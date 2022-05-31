"""
Script that can be edited to configure and run a queue of training runs.
Must be run as main

See the actual training function in train.py
"""

import comet_ml  # Required first for auto-logging

import importlib  # to reload config.py between each run
import numpy as np
import copy
import time
import gc

import torch

import config
import train
import utils.exception


# TODO intercept Ctrl-C sigint and ask for confirmation



""" Global config modifications (applied to all runs) """
train_all_k_folds = False  # automatically train all cross-validation folds?
plot_period = 40  # when launching lots of runs: don't plot too much (approx. 500MB of comet.ml data / run...)
plot_epoch_0 = False

"""
Please write two lists of dicts, such that:
- (model|train)_config_mods contains the modifications applied to config.model and config.train, resp.
- each list index corresponds to a training run
- each dict key corresponds to an attribute of config.model.* or config.train.*. Empty dict to indicate
      that no config modification should be performed
"""
model_config_mods, train_config_mods = list(), list()


#for label_smoothing in [0.0, 0.1, 0.5]:
#for beta in np.flip(np.logspace(-4.0, -2.0, num=5, endpoint=False)):  # Powers of 10.0
for beta in [1.55e-5]:
    model_config_mods.append({'run_name': 'beta{:.1e}'.format(beta)})
    train_config_mods.append({'beta': beta})



if __name__ == "__main__":
    assert len(model_config_mods) == len(train_config_mods)

    base_model_config, base_train_config = config.ModelConfig(), config.TrainConfig()

    # If performing k-fold cross validation trains, duplicate run mods to train all folds
    if train_all_k_folds:
        model_config_mods_kfolds, train_config_mods_kfolds = list(), list()
        for base_run_index in range(len(model_config_mods)):
            for fold_idx in range(base_train_config.k_folds):
                # duplicate this run configuration, one duplicate per fold
                model_config_mods_kfolds.append(copy.deepcopy(model_config_mods[base_run_index]))
                train_config_mods_kfolds.append(copy.deepcopy(train_config_mods[base_run_index]))
                train_config_mods_kfolds[-1]['current_k_fold'] = fold_idx
                # k-fold index appended to the name
                if 'run_name' in model_config_mods[base_run_index]:
                    run_name = model_config_mods[base_run_index]['run_name'] + '_kf{}'.format(fold_idx)
                else:
                    run_name = base_model_config.run_name + '_kf{}'.format(fold_idx)
                model_config_mods_kfolds[-1]['run_name'] = run_name
        model_config_mods, train_config_mods = model_config_mods_kfolds, train_config_mods_kfolds
    del base_model_config, base_train_config

    # = = = = = = = = = = Training queue: main loop = = = = = = = = = =
    for run_index in range(len(model_config_mods)):
        model_config, train_config = config.ModelConfig(), config.TrainConfig()  # Start from defaults from config.py

        print("================================================================")
        print("=============== Enqueued Training Run {}/{} starts ==============="
              .format(run_index+1, len(model_config_mods)))

        # Modifications applied to all runs
        train_config.plot_epoch_0 = False
        train_config.plot_period = plot_period
        # Per-run config modifications
        for k, v in model_config_mods[run_index].items():
            model_config.__dict__[k] = v
        for k, v in train_config_mods[run_index].items():
            train_config.__dict__[k] = v
        # Required before any actual train - after hparams have been properly set
        config.update_dynamic_config_params(model_config, train_config)

        # Model train. An occasional model divergence (sometimes happen during first epochs) is tolerated
        #    Full-AR Normalizing Flows (e.g. MAF/IAF) are very unstable and hard to train on dim>100 latent spaces
        max_divergent_model_runs = 2  # 2 diverging runs are already a lot... a 3rd diverging run stops training
        divergent_model_runs = 0
        has_finished_training = False
        while not has_finished_training:
            try:  # - - - - - Model train - - - - -
                train.train_model(model_config, train_config)
                has_finished_training = True
            except utils.exception.ModelConvergenceError as e:
                divergent_model_runs += 1
                if divergent_model_runs <= max_divergent_model_runs:
                    print("[train_queue.py] Model train did not converge: {}. Restarting run... (next trial: {}/{})"
                          .format(e, divergent_model_runs + 1, max_divergent_model_runs + 1))
                    model_config.allow_erase_run = True  # We force the run to be erasable
                else:
                    e_str = "Model training run {}/{} does not converge ({} run trials failed). " \
                            "Training queue will now stop, please check this convergence problem."\
                        .format(run_index+1, len(model_config_mods), divergent_model_runs)
                    raise utils.exception.ModelConvergenceError(e_str)

        print("=============== Enqueued Training Run {}/{} has finished ==============="
              .format(run_index+1, len(model_config_mods)))
        print("======================================================================")

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        # Maybe PyTorch / Python GC need some time to empty CUDA buffers...
        # The out-of-memory crash now happens very rarely, but remains quite unexplained
        time.sleep(20)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

