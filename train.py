"""
This script performs a single training run for the configuration described
in config.py, when running as __main__.

Its train_config(...) function can also be called from another script,
with small modifications to the config (enqueued train runs).

See train_queue.py for enqueued training runs
"""
import copy

import comet_ml  # Required first for auto-logging

import gc
from pathlib import Path
from typing import Optional, Dict, List, Union

import numpy as np
import mkl
import torch
import torch.nn as nn
import torch.optim
import torch.profiler

import config
import evalinterp
import model.base
import model.hierarchicalvae
import logs.logger
import logs.metrics
from logs.metrics import SimpleMetric, EpochMetric, VectorMetric, LatentMetric, LatentCorrMetric
import data.dataset
import data.build
import utils.profiling
from utils.hparams import LinearDynamicParam
import utils.figures
import utils.exception



def train_model(model_config: config.ModelConfig, train_config: config.TrainConfig):
    """ Performs a full training run, as described by parameters in config.py.

    Some attributes from config.py might be dynamically changed by train_queue.py (or this script,
    after loading the datasets) - so they can be different from what's currently written in config.py. """

    torch.manual_seed(0)

    # ========== Logger init (required for comet.ml console logs, load from checkpoint, ...) and Config check ==========
    root_path = Path(__file__).resolve().parent
    logger = logs.logger.RunLogger(root_path, model_config, train_config)


    # ========== Datasets and DataLoaders ==========
    pretrain_audio = train_config.pretrain_audio_only  # type: bool
    if pretrain_audio:
        train_audio_dataset, validation_audio_dataset = data.build.get_pretrain_datasets(model_config, train_config)
        # dataloader is a dict of 2 dataloaders ('train' and 'validation')
        dataloader, dataloaders_nb_items = data.build.get_pretrain_dataloaders(
            model_config, train_config, train_audio_dataset, validation_audio_dataset)
        preset_helper = None
    else:
        # Must be constructed first because dataset output sizes will be required to automatically
        # infer models output sizes.
        dataset = data.build.get_dataset(model_config, train_config)
        # We use a single dataset (for train, valid, test) but different dataloaders
        train_audio_dataset, validation_audio_dataset = dataset, dataset
        preset_helper = dataset.preset_indexes_helper
        # dataloader is a dict of 3 subsets dataloaders ('train', 'validation' and 'test')
        # This function will make copies of the original dataset (some with, some without data augmentation)
        dataloader, dataloaders_nb_items = data.build.get_split_dataloaders(train_config, dataset)


    # ================== Model definition - includes Losses, Optimizers, Schedulers ===================
    #                            (requires the full_dataset to be built)
    # HierarchicalVAE won't be the same during pre-train (empty parts without optimizer and scheduler)
    ae_model = model.hierarchicalvae.HierarchicalVAE(model_config, train_config, preset_helper)
    # will torchinfo txt summary. model must not be parallel (graph not written anymore: too complicated, unreadable)
    logger.init_with_model(ae_model, model_config.input_audio_tensor_size, write_graph=False)


    # ============================= Training devices (GPU(s) only) =========================
    if train_config.verbosity >= 1:
        print("Intel MKL num threads = {}. PyTorch num threads = {}. CUDA devices count: {} GPU(s)."
              .format(mkl.get_max_threads(), torch.get_num_threads(), torch.cuda.device_count()))
    if torch.cuda.device_count() == 0:
        raise NotImplementedError()  # CPU training not available
    elif torch.cuda.device_count() == 1:
        device = 'cuda:0'
        parallel_device_ids = [0]  # "Parallel" 1-GPU model
    else:
        device = torch.device('cuda:{}'.format(train_config.main_cuda_device_idx))
        # We use all available GPUs - the main one must be first in list
        parallel_device_ids = [i for i in range(torch.cuda.device_count()) if i != train_config.main_cuda_device_idx]
        parallel_device_ids.insert(0, train_config.main_cuda_device_idx)
    ae_model_parallel = nn.DataParallel(ae_model, device_ids=parallel_device_ids, output_device=device)


    # ========== Load weights from pre-trained models? ==========
    if not pretrain_audio:
        if model_config.pretrained_VAE_checkpoint is not None:
            ae_model.load_checkpoints(model_config.pretrained_VAE_checkpoint)  # FIXME specify device for map_location
        else:
            if train_config.verbosity >= 1:
                print("Training starts from scratch (model_config.pretrained_VAE_checkpoint is None).")


    # ========== Scalars, metrics, images and audio to be tracked in Tensorboard ==========
    # Some of these metrics might be unused during pre-training
    # Special 'super-metrics', used by 1D scalars or metrics to retrieve stored data. Not directly logged
    super_metrics = {
        'LatentMetric/Train': LatentMetric(model_config.dim_z, dataloaders_nb_items['train'],
                                           dim_label=train_audio_dataset.available_labels_count),
        'LatentMetric/Valid': LatentMetric(model_config.dim_z, dataloaders_nb_items['validation'],
                                           dim_label=validation_audio_dataset.available_labels_count),
        'RegOutValues/Train': VectorMetric(dataloaders_nb_items['train']),
        'RegOutValues/Valid': VectorMetric(dataloaders_nb_items['validation'])
    }
    # 1D scalars with a .get() method. All of these will be automatically added to Tensorboard
    scalars = {  # Audio reconstruction negative log prob + monitoring metrics comparable across all models
        'Audio/LogProbLoss/Train': EpochMetric(), 'Audio/LogProbLoss/Valid': EpochMetric(),
        'Audio/MSE/Train': EpochMetric(), 'Audio/MSE/Valid': EpochMetric(),
        # 'AudioLoss/SC/Train': EpochMetric(), 'AudioLoss/SC/Valid': EpochMetric(),  # TODO
        # Latent-space and VAE losses
        'Latent/Loss/Train': EpochMetric(), 'Latent/Loss/Valid': EpochMetric(),  # without beta
        'Latent/BackpropLoss/Train': EpochMetric(), 'Latent/BackpropLoss/Valid': EpochMetric(),
        'Latent/MMD/Train': EpochMetric(), 'Latent/MMD/Valid': EpochMetric(),
        'Latent/MaxAbsVal/Train': SimpleMetric(), 'Latent/MaxAbsVal/Valid': SimpleMetric(),
        'Latent/AlignLoss/Train': EpochMetric(), 'Latent/AlignLoss/Valid': EpochMetric(),  # for aligned VAEs only
        'VAELoss/Total/Train': EpochMetric(), 'VAELoss/Total/Valid': EpochMetric(),
        'VAELoss/Backprop/Train': EpochMetric(), 'VAELoss/Backprop/Valid': EpochMetric(),
        # Presets (synth controls) losses used for backprop
        #        + monitoring metrics (numerical L1 loss, categorical accuracy)
        # Preset/... correspond to Automatic Synthesizer Programming metrics (preset inferred from audio)
        'Preset/NLL/Total/Train': EpochMetric(), 'Preset/NLL/Total/Valid': EpochMetric(),
        'Preset/NLL/Numerical/Train': EpochMetric(), 'Preset/NLL/Numerical/Valid': EpochMetric(),
        'Preset/NLL/CatCE/Train': EpochMetric(), 'Preset/NLL/CatCE/Valid': EpochMetric(),
        #'Controls/RegulLoss/Train': EpochMetric(), 'Controls/RegulLoss/Valid': EpochMetric(),
        'Preset/L1error/Train': EpochMetric(), 'Preset/L1error/Valid': EpochMetric(),
        'Preset/Accuracy/Train': EpochMetric(), 'Preset/Accuracy/Valid': EpochMetric(),
        # 'Dynamic' scheduling hyper-params
        'Sched/LRwarmup': LinearDynamicParam(
            train_config.lr_warmup_start_factor, 1.0,
            end_epoch=train_config.lr_warmup_epochs, current_epoch=train_config.start_epoch),
        'Sched/AttGamma': LinearDynamicParam(
            0.0, model_config.attention_gamma,
            end_epoch=(train_config.attention_gamma_warmup_period if pretrain_audio else 0)),
        'Sched/VAE/beta': LinearDynamicParam(train_config.beta_start_value, train_config.beta,
                                             end_epoch=train_config.beta_warmup_epochs,
                                             current_epoch=train_config.start_epoch),
        'Sched/PresetDec/SamplingP': LinearDynamicParam(
            0.0, train_config.preset_sched_sampling_max_p, end_epoch=train_config.preset_sched_sampling_warmup_epochs)
    }
    for k in ae_model.trained_param_group_names:
        scalars['Sched/{}/LR'.format(k)] = SimpleMetric(train_config.initial_learning_rate[k])
    # All hparams are supposed to be set (e.g. automatic dim_z) at this point, and can be logged
    logger.log_hyper_parameters()


    # ========== PyTorch Profiling (optional) ==========
    optional_profiler = utils.profiling.OptionalProfiler(train_config, logger.tensorboard_run_dir)


    # ========== Model training epochs ==========
    for epoch in range(train_config.start_epoch, train_config.n_epochs):
        # = = = = = Re-init of epoch metrics and useful scalars (warmup ramps, ...) = = = = =
        logger.on_epoch_starts(epoch, scalars, super_metrics)

        # = = = = = Warmups = = = = =
        if epoch <= train_config.lr_warmup_epochs:  # LR warmups bypass the schedulers during first epochs
            ae_model.set_warmup_lr_factor(scalars['Sched/LRwarmup'].get(epoch))
        # ae_model.set_attention_gamma(scalars['Sched/AttGamma'].get(epoch))  # FIXME re-activate after implemented
        ae_model.set_preset_decoder_scheduled_sampling_p(scalars['Sched/PresetDec/SamplingP'].get(epoch))
        ae_model.beta_warmup_ongoing = not scalars['Sched/VAE/beta'].has_reached_final_value

        # = = = = = Train all mini-batches (optional profiling) = = = = =
        #torch.autograd.set_detect_anomaly(True)  # FIXME
        # when profiling is disabled: true no-op context manager, and prof is None
        with optional_profiler.get_prof(epoch) as prof:  # TODO use comet context if available
            ae_model_parallel.train()
            dataloader_iter = iter(dataloader['train'])
            for i in range(len(dataloader['train'])):
                minibatch = next(dataloader_iter)
                x_in, v_in, uid, notes, label = [m.to(device) for m in minibatch]
                model.hierarchicalvae.process_minibatch(
                    ae_model, ae_model_parallel, device,
                    x_in, v_in, uid, notes, label,
                    epoch, scalars, super_metrics
                )
                # End of mini-batch (step)
                logger.on_train_minibatch_finished(i)
                if prof is not None:
                    prof.step()
        scalars['Latent/MaxAbsVal/Train'].set(np.abs(super_metrics['LatentMetric/Train'].get_z('zK')).max())

        # = = = = = Evaluation on validation dataset (no profiling) = = = = =
        if logger.should_validate:  # don't always compute validation (very long w/ transformer AR decoders)
            with torch.no_grad():
                ae_model_parallel.eval()  # BN stops running estimates, Transformers use AR eval mode
                v_out_backup, v_in_backup = [], []  # Params inference error (Comet/Tensorboard plot)
                i_to_plot = np.random.default_rng(seed=epoch).integers(0, len(dataloader['validation'])-1)
                for i, minibatch in enumerate(dataloader['validation']):
                    x_in, v_in, uid, notes, label = [m.to(device) for m in minibatch]
                    ae_out_audio, ae_out_preset = model.hierarchicalvae.process_minibatch(
                        ae_model, ae_model_parallel, device,
                        x_in, v_in, uid, notes, label,
                        epoch, scalars, super_metrics
                    )
                    # Validation plots
                    if logger.should_plot:
                        v_in_backup.append(v_in)  # Full-batch error storage - will be used later
                        v_out_backup.append(ae_out_preset.u_out)
                        if ae_out_audio is not None:
                            if i == i_to_plot:  # random mini-batch plot (validation dataset is not randomized)
                                logger.plot_spectrograms(
                                    x_in, ae_out_audio.x_sampled, uid, notes, validation_audio_dataset)
                                logger.plot_decoder_interpolation(
                                    ae_model, ae_model.flatten_latent_values(ae_out_audio.z_sampled),
                                    uid, validation_audio_dataset, audio_channel=model_config.main_midi_note_index)
            scalars['Latent/MaxAbsVal/Valid'].set(np.abs(super_metrics['LatentMetric/Valid'].get_z('zK')).max())

        ae_model.schedulers_step()
        for k in ae_model.trained_param_group_names:
            scalars['Sched/' + k + '/LR'].set(ae_model.get_group_lr(k))
        # Possible early stop if reg model is not learning anything anymore
        # early_stop = (reg_model.learning_rate < train_config.early_stop_lr_threshold['reg'])
        early_stop = False  # FIXME re-implement early stop

        # = = = = = Epoch logs (scalars/sounds/images + updated metrics) = = = = =
        logger.add_scalars(scalars)  # Some scalars might not be added (e.g. during pretrain, if no validation, ...)
        if logger.should_plot or early_stop:
            logger.plot_stats__threaded(super_metrics, ae_model, validation_audio_dataset)  # non-blocking
            # TODO also thread this one (30s plot... !!!!)
            if len(v_in_backup) > 0 and not pretrain_audio:
                v_in_backup, v_out_backup = torch.cat(v_in_backup), torch.cat(v_out_backup)
                fig, _ = utils.figures.plot_preset2d_batch_error(
                    v_out_backup.detach().cpu(), v_in_backup.detach().cpu(), preset_helper)
                logger.add_figure('PresetError/Validation', fig)

        # = = = = = End of epoch = = = = =
        logger.on_epoch_finished(epoch)
        if early_stop:
            print("[train.py] Training stopped early (final loss plateau)")
            break


    # ========== Logger final stats + save Model/Optimizers/Scheduler checkpoints ==========
    ae_model.save_checkpoints(logger.run_dir)
    logger.on_training_finished()  # Might have to wait for threads


    # ========== "Manual GC" (to try to prevent random CUDA out-of-memory between enqueued runs) ==========
    # This should not have any effect by Python is supposed to store objects as reference-counted, but this
    # seems to be effective... maybe because pytorch keeps some internal references, and some CUDA tensors stay alive?
    del ae_model_parallel, ae_model, ae_out_audio, ae_out_preset
    locals_names = copy.deepcopy(list(locals().keys()))
    for k in locals_names:  # get all local variables, delete any tensor variable
        if not k.startswith('_') and isinstance(locals()[k], torch.Tensor):
            del locals()[k]
    del scalars, super_metrics, logger
    del dataloader, dataloader_iter
    del train_audio_dataset, validation_audio_dataset
    try:
        del dataset  # May not exist
    except UnboundLocalError:
        pass
    gc.collect()


    # ========== Evaluate interpolation performance ==========
    # Parallel, will use saved checkpoint
    if train_config.evaluate_interpolation_after_training:
        evalinterp.eval_single_model("{}/{}".format(model_config.name, model_config.run_name))




if __name__ == "__main__":
    # Normal run, current values from config.py will be used to parametrize learning and models
    _model_config, _train_config = config.ModelConfig(), config.TrainConfig()
    config.update_dynamic_config_params(_model_config, _train_config)  # Required before any actual train
    train_model(_model_config, _train_config)

