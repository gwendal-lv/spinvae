"""
This script performs a single training run for the configuration described
in config.py, when running as __main__.

Its train_config(...) function can also be called from another script,
with small modifications to the config (enqueued train runs).

See train_queue.py for enqueued training runs
"""
import copy

import comet_ml  # Required first for auto-logging

import multiprocessing
import gc
from pathlib import Path
import contextlib
from typing import Optional, Dict, List

import numpy as np
import mkl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim
import torch.profiler

import config
import model.base
import model.loss
import model.build
import model.extendedAE
import model.flows
import model.hierarchicalvae
import logs.logger
import logs.metrics
from logs.metrics import SimpleMetric, EpochMetric, VectorMetric, LatentMetric, LatentCorrMetric
import data.dataset
import data.build
import utils.configutils
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
            ae_model.load_checkpoints(model_config.pretrained_VAE_checkpoint)
        else:
            if train_config.verbosity >= 1:
                print("Training starts from scratch (model_config.pretrained_VAE_checkpoint is None).")


    # ========== Scalars, metrics, images and audio to be tracked in Tensorboard ==========
    # Some of these metrics might be unused during pre-training
    # Special 'super-metrics', used by 1D scalars or metrics to retrieve stored data. Not directly logged
    super_metrics = {'LatentMetric/Train': LatentMetric(model_config.dim_z, dataloaders_nb_items['train'],
                                                        dim_label=train_audio_dataset.available_labels_count),
                     'LatentMetric/Valid': LatentMetric(model_config.dim_z, dataloaders_nb_items['validation'],
                                                        dim_label=validation_audio_dataset.available_labels_count),
                     'RegOutValues/Train': VectorMetric(dataloaders_nb_items['train']),
                     'RegOutValues/Valid': VectorMetric(dataloaders_nb_items['validation'])}
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
               'VAELoss/Total/Train': EpochMetric(), 'VAELoss/Total/Valid': EpochMetric(),
               'VAELoss/Backprop/Train': EpochMetric(), 'VAELoss/Backprop/Valid': EpochMetric(),
               # Presets (synth controls) losses used for backprop
               #        + monitoring metrics (numerical L1 loss, categorical accuracy)
               'Preset/NLL/Total/Train': EpochMetric(), 'Preset/NLL/Total/Valid': EpochMetric(),
               'Preset/NLL/Numerical/Train': EpochMetric(), 'Preset/NLL/Numerical/Valid': EpochMetric(),
               'Preset/NLL/CatCE/Train': EpochMetric(), 'Preset/NLL/CatCE/Valid': EpochMetric(),
               #'Controls/RegulLoss/Train': EpochMetric(), 'Controls/RegulLoss/Valid': EpochMetric(),
               'Preset/L1error/Train': EpochMetric(), 'Preset/L1error/Valid': EpochMetric(),  # FIXME rename QError
               'Preset/Accuracy/Train': EpochMetric(), 'Preset/Accuracy/Valid': EpochMetric(),
               # Other misc. metrics
               'Sched/LRwarmup': LinearDynamicParam(
                   train_config.lr_warmup_start_factor, 1.0,
                   end_epoch=train_config.lr_warmup_epochs, current_epoch=train_config.start_epoch),
               'Sched/AttGamma': LinearDynamicParam(
                   0.0, model_config.attention_gamma,
                   end_epoch=(train_config.attention_gamma_warmup_period if pretrain_audio else 0)),
               'Sched/VAE/beta': LinearDynamicParam(train_config.beta_start_value, train_config.beta,
                                                    end_epoch=train_config.beta_warmup_epochs,
                                                    current_epoch=train_config.start_epoch) }
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
        ae_model.beta_warmup_ongoing = not scalars['Sched/VAE/beta'].has_reached_final_value

        # = = = = = Train all mini-batches (optional profiling) = = = = =
        # when profiling is disabled: true no-op context manager, and prof is None
        with optional_profiler.get_prof(epoch) as prof:  # TODO use comet context if available
            ae_model_parallel.train()
            dataloader_iter = iter(dataloader['train'])
            for i in range(len(dataloader['train'])):
                minibatch = next(dataloader_iter)
                x_in, v_in, uid, notes, label = [m.to(device) for m in minibatch]
                # reg_model.precompute_u_in_permutations(v_in)  # FIXME pre-compute permutations
                ae_model.optimizers_zero_grad()
                ae_out = ae_model_parallel(x_in, v_in, uid, notes)  # TODO auto-encode presets
                ae_out = ae_model.parse_outputs(ae_out)
                # v_out = reg_model_parallel(ae_out.z_sampled[0])  # FIXME don't use reg_model anymore
                # reg_model.precompute_u_out_with_symmetries(v_out)   # FIXME pre-compute permutations
                super_metrics['LatentMetric/Train'].append_hierarchical_latent(ae_out, label)
                # Losses (computed on 1 GPU using the non-parallel original model instance)
                audio_log_prob_loss = ae_model.decoder.audio_log_prob_loss(ae_out.x_decoded_proba, x_in)
                scalars['Audio/LogProbLoss/Train'].append(audio_log_prob_loss)
                lat_loss, lat_backprop_loss = ae_model.latent_loss(ae_out, scalars['Sched/VAE/beta'].get(epoch))
                scalars['Latent/Loss/Train'].append(lat_loss)
                scalars['Latent/BackpropLoss/Train'].append(lat_backprop_loss)  # Includes beta
                extra_lat_reg_loss = 0.0  # Can be used for extra regularisation, contrastive loss...
                extra_lat_reg_loss *= scalars['Sched/VAE/beta'].get(epoch)
                if not pretrain_audio:
                    u_categorical_nll, u_numerical_nll = ae_out.u_categorical_nll.mean(), ae_out.u_numerical_nll.mean()
                    preset_loss = u_categorical_nll + u_numerical_nll
                    scalars['Preset/NLL/Total/Train'].append(preset_loss)
                    preset_loss *= train_config.params_loss_compensation_factor
                    scalars['Preset/NLL/Numerical/Train'].append(u_numerical_nll)
                    scalars['Preset/NLL/CatCE/Train'].append(u_categorical_nll)
                else:
                    preset_loss = torch.zeros((1,), device=device)
                preset_reg_loss = torch.zeros((1,), device=device)  # No regularization yet....
                with torch.no_grad():  # Monitoring-only losses
                    scalars['Audio/MSE/Train'].append(F.mse_loss(ae_out.x_sampled, x_in))
                    scalars['Latent/MMD/Train'].append(ae_model.mmd(ae_out.get_z_sampled_no_hierarchy()))
                    if not pretrain_audio:
                        scalars['Preset/Accuracy/Train'].append(ae_out.u_accuracy.mean())
                        scalars['Preset/L1error/Train'].append(ae_out.u_l1_error.mean())
                    scalars['VAELoss/Total/Train'].append(ae_model.vae_loss(audio_log_prob_loss, x_in.shape, ae_out))
                    scalars['VAELoss/Backprop/Train'].append(
                        audio_log_prob_loss + lat_backprop_loss + extra_lat_reg_loss)
                utils.exception.check_nan_values(
                    epoch, audio_log_prob_loss, lat_backprop_loss, extra_lat_reg_loss, preset_loss, preset_reg_loss)
                # Backprop and optimizers' step (before schedulers' step)
                (audio_log_prob_loss + lat_backprop_loss + extra_lat_reg_loss + preset_loss + preset_reg_loss).backward()
                ae_model.optimizers_step()
                # End of mini-batch (step)
                logger.on_train_minibatch_finished(i)
                if prof is not None:
                    prof.step()
        scalars['Latent/MaxAbsVal/Train'].set(np.abs(super_metrics['LatentMetric/Train'].get_z('zK')).max())

        # = = = = = Evaluation on validation dataset (no profiling) = = = = =
        with torch.no_grad():  # TODO use comet context if available
            ae_model_parallel.eval()  # BN stops running estimates
            v_out_backup, v_in_backup = [], []  # Params inference error (Comet/Tensorboard plot)
            i_to_plot = np.random.default_rng(seed=epoch).integers(0, len(dataloader['validation'])-1)
            for i, minibatch in enumerate(dataloader['validation']):
                x_in, v_in, uid, notes, label = [m.to(device) for m in minibatch]
                # reg_model.precompute_u_in_permutations(v_in)  # FIXME pre-compute permutations
                ae_out = ae_model_parallel(x_in, v_in, uid, notes)  # TODO auto-encode presets
                ae_out = ae_model.parse_outputs(ae_out)
                # v_out = reg_model_parallel(ae_out.z_sampled[0])  # FIXME don't use reg_model anymore
                # reg_model.precompute_u_out_with_symmetries(v_out)  # FIXME pre-compute permutations
                super_metrics['LatentMetric/Valid'].append_hierarchical_latent(ae_out, label)
                audio_log_prob_loss = ae_model.decoder.audio_log_prob_loss(ae_out.x_decoded_proba, x_in)
                scalars['Audio/LogProbLoss/Valid'].append(audio_log_prob_loss)
                lat_loss, lat_backprop_loss = ae_model.latent_loss(ae_out, scalars['Sched/VAE/beta'].get(epoch))
                scalars['Latent/Loss/Valid'].append(lat_loss)
                scalars['Latent/BackpropLoss/Valid'].append(lat_backprop_loss)
                scalars['Audio/MSE/Valid'].append(F.mse_loss(ae_out.x_sampled, x_in))
                scalars['Latent/MMD/Valid'].append(ae_model.mmd(ae_out.get_z_sampled_no_hierarchy()))
                if not pretrain_audio:
                    u_categorical_nll, u_numerical_nll = ae_out.u_categorical_nll.mean(), ae_out.u_numerical_nll.mean()
                    preset_loss = u_categorical_nll + u_numerical_nll
                    scalars['Preset/NLL/Total/Valid'].append(preset_loss)
                    preset_loss *= train_config.params_loss_compensation_factor
                    scalars['Preset/NLL/Numerical/Valid'].append(u_numerical_nll)
                    scalars['Preset/NLL/CatCE/Valid'].append(u_categorical_nll)
                    scalars['Preset/Accuracy/Valid'].append(ae_out.u_accuracy.mean())
                    scalars['Preset/L1error/Valid'].append(ae_out.u_l1_error.mean())
                scalars['VAELoss/Total/Valid'].append(ae_model.vae_loss(audio_log_prob_loss, x_in.shape, ae_out))
                scalars['VAELoss/Backprop/Valid'].append(audio_log_prob_loss + lat_backprop_loss)
                # Validation plots
                if logger.should_plot:
                    v_in_backup.append(v_in)  # Full-batch error storage - will be used later
                    v_out_backup.append(ae_out.u_out)
                    if i == i_to_plot:  # random mini-batch plot (validation dataset is not randomized)
                        logger.plot_spectrograms(x_in, ae_out.x_sampled, uid, notes, validation_audio_dataset)
                        logger.plot_decoder_interpolation(
                            ae_model, ae_model.flatten_latent_values(ae_out.z_sampled),
                            uid, validation_audio_dataset, audio_channel=model_config.main_midi_note_index)
        scalars['Latent/MaxAbsVal/Valid'].set(np.abs(super_metrics['LatentMetric/Valid'].get_z('zK')).max())

        # Dynamic LR scheduling depends on validation performance. Losses for plateau-detection are chosen in config.py
        ae_model.schedulers_step(
            {k: scalars[train_config.scheduler_losses[k] + '/Valid'].get() for k in ae_model.trained_param_group_names})
        for k in ae_model.trained_param_group_names:
            scalars['Sched/' + k + '/LR'].set(ae_model.get_group_lr(k))
        # Possible early stop if reg model is not learning anything anymore
        # early_stop = (reg_model.learning_rate < train_config.early_stop_lr_threshold['reg'])
        early_stop = False  # FIXME re-implement early stop

        # = = = = = Epoch logs (scalars/sounds/images + updated metrics) = = = = =
        logger.add_scalars(scalars)  # Some scalars might not be added (e.g. during pretrain)
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
    del ae_model_parallel, ae_model, ae_out
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


if __name__ == "__main__":
    # Normal run, current values from config.py will be used to parametrize learning and models
    _model_config, _train_config = config.ModelConfig(), config.TrainConfig()
    config.update_dynamic_config_params(_model_config, _train_config)  # Required before any actual train
    train_model(_model_config, _train_config)

