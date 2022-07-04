"""
Script that can be edited to configure and run a queue of training runs.
Must be run as main

See the actual training function in train.py
"""
import pathlib

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

"""
This folder and sub-folders can be entirely duplicated before running this (duplicated) train_queue.py file.
This allows to work on the original python code while the training queue (which may last for days)
runs using isolated duplicated python code.
"""
duplicate_code = True

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



# Train arrêté à hvae/8x2_dimz100_lvls1_big_att_lat2l_k3x3__dklgammaFalse (trop long...)
# c'était le 7/24 de la série des "attention"
i = 0
for dim_z in [100]:
    for conv_layers in ['8x2']:  # '8x1' already done
        for big in ['_big_att', '_att']:
            for latent_arch in ['conv_1l_k1x1_gated', 'conv_2l_k3x3_gated']:
                for num_levels in [1, 2, 3]:
                    for dkl_auto_gamma in [False, True]:
                        i += 1
                        if i <= 7:  # entraînements déjà faits dans cette série
                            continue

                        conv_arch = 'specladder' + conv_layers + '_res' + big
                        model_config_mods.append(
                            {'comet_tags': ['conv_hpar_sweep'],
                             'run_name': '{}_dimz{}_lvls{}{}_lat{}__dklgamma{}'
                                 .format(conv_layers, dim_z, num_levels, big, latent_arch[5:12], dkl_auto_gamma),
                             'vae_latent_levels': num_levels,
                             'approx_requested_dim_z': dim_z,
                             'vae_latent_extract_architecture': latent_arch,
                             'vae_main_conv_architecture': conv_arch,

                             }
                        )
                        train_config_mods.append({'dkl_auto_gamma': dkl_auto_gamma})




if __name__ == "__main__":
    if duplicate_code:
        import sys
        import subprocess
        import utils.code
        argv = sys.argv
        # Check if we're the original train_queue.py, or if we're the duplicated code run from a fork
        if '--DUPLICATED-CODE' not in argv:
            code_dir = pathlib.Path(__file__).parent
            duplicated_code_dir = code_dir.joinpath("DUPLICATED_CODE")
            print("[[ORIGINAL train_queue.py]] All contents from this directory (including hidden files, ...) "
                  "will be copied to {}".format(duplicated_code_dir))
            utils.code.duplicate_code(code_dir, duplicated_code_dir, ['saved', 'exports', 'figures'])
            print("[[ORIGINAL train_queue.py]] File copy finished. Running the duplicated code now...")
            # Use the current python executable
            subprocess.run([sys.executable, str(duplicated_code_dir.joinpath('train_queue.py')), '--DUPLICATED-CODE'])
            print("[[ORIGINAL train_queue.py]] Duplicated code finished - exiting now...")
            sys.exit(0)
        else:
            print("[[DUPLICATED FORKED train_queue.py]] Training queue is about to start...")

    base_model_config, base_train_config = config.ModelConfig(), config.TrainConfig()

    assert len(model_config_mods) == len(train_config_mods)

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
        train_config.plot_epoch_0 = plot_epoch_0
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

    print("[train_queue.py] Finished training.")
