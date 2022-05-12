"""
This script performs a single training run for the configuration described
in config.py, when running as __main__.

Its train_config(...) function can also be called from another script,
with small modifications to the config (enqueued train runs).

See train_queue.py for enqueued training runs
"""
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
import logs.logger
import logs.metrics
from logs.metrics import SimpleMetric, EpochMetric, VectorMetric, LatentMetric, LatentCorrMetric
import data.dataset
import data.build
import utils.profile
from utils.hparams import LinearDynamicParam
import utils.figures
import utils.exception


def train_config():
    """ Performs a full training run, as described by parameters in config.py.

    Some attributes from config.py might be dynamically changed by train_queue.py (or this script,
    after loading the datasets) - so they can be different from what's currently written in config.py. """
    torch.manual_seed(0)


    # ========== Datasets and DataLoaders ==========
    pretrain_vae = config.train.pretrain_ae_only  # type: bool
    if pretrain_vae:
        train_audio_dataset, validation_audio_dataset = data.build.get_pretrain_datasets(config.model, config.train)
        dataset = None
        # dataloader is a dict of 2 dataloaders ('train' and 'validation')
        dataloader, dataloaders_nb_items = data.build.get_pretrain_dataloaders(config.model, config.train,
                                                                               train_audio_dataset,
                                                                               validation_audio_dataset)
        preset_indexes_helper = None
    else:
        # Must be constructed first because dataset output sizes will be required to automatically
        # infer models output sizes.
        dataset = data.build.get_dataset(config.model, config.train)
        train_audio_dataset, validation_audio_dataset = None, None
        preset_indexes_helper = dataset.preset_indexes_helper
        # dataloader is a dict of 3 subsets dataloaders ('train', 'validation' and 'test')
        # This function will make copies of the original dataset (some with, some without data augmentation)
        dataloader, dataloaders_nb_items = data.build.get_split_dataloaders(config.train, dataset)


    # ========== Logger init (required to load from checkpoint) and Config check ==========
    root_path = Path(__file__).resolve().parent
    logger = logs.logger.RunLogger(root_path, config.model, config.train)
    if logger.restart_from_checkpoint:
        model.build.check_configs_on_resume_from_checkpoint(config.model, config.train,
                                                            logger.get_previous_config_from_json())


    # ========== Model definition (requires the full_dataset to be built) + Losses included in models ==========
    # The extended_ae_model has all sub-models as attributes (even if some sub-models, e.g. synth controls regression,
    # are set to None during pre-training). Useful to change device or train/eval status of all models.
    if pretrain_vae:
        _, _, ae_model = model.build.build_ae_model(config.model, config.train)
        reg_model = model.base.DummyRegModel()
        extended_ae_model = model.extendedAE.ExtendedAE(ae_model, reg_model)
    else:
        _, _, ae_model, reg_model, extended_ae_model = model.build.build_extended_ae_model(config.model, config.train,
                                                                                           preset_indexes_helper)
    extended_ae_model.eval()
    # will torchinfo txt summary. model must not be parallel (graph not written anymore: too complicated, unreadable)
    logger.init_with_model(ae_model, config.model.input_tensor_size, write_graph=False)  # main model: autoencoder
    if not isinstance(reg_model, model.base.DummyModel):
        logger.write_model_summary(reg_model, (config.train.minibatch_size, config.model.dim_z), "reg")  # Other model


    # ========== Training devices (GPU(s) only) ==========
    if config.train.verbosity >= 1:
        print("Intel MKL num threads = {}. PyTorch num threads = {}. CUDA devices count: {} GPU(s)."
              .format(mkl.get_max_threads(), torch.get_num_threads(), torch.cuda.device_count()))
    if torch.cuda.device_count() == 0:
        raise NotImplementedError()  # CPU training not available
    elif torch.cuda.device_count() == 1:
        device = 'cuda:0'
        parallel_device_ids = [0]  # "Parallel" 1-GPU model
    else:
        device = torch.device('cuda:{}'.format(config.train.main_cuda_device_idx))
        # We use all available GPUs - the main one must be first in list
        parallel_device_ids = [i for i in range(torch.cuda.device_count()) if i != config.train.main_cuda_device_idx]
        parallel_device_ids.insert(0, config.train.main_cuda_device_idx)
    extended_ae_model = extended_ae_model.to(device)
    ae_model_parallel = nn.DataParallel(ae_model, device_ids=parallel_device_ids, output_device=device)
    reg_model_parallel = nn.DataParallel(reg_model, device_ids=parallel_device_ids, output_device=device)


    # ========== Optimizer and Scheduler ==========
    ae_model.init_optimizer_and_scheduler()
    reg_model.init_optimizer_and_scheduler()


    # ========== Restart from checkpoint, load weights from pre-trained models? ==========
    if pretrain_vae:
        if logger.restart_from_checkpoint:
            start_checkpoint = logs.logger.get_model_checkpoint(root_path, config.model, config.train.start_epoch - 1)
            ae_model.load_checkpoint(start_checkpoint)
    else:
        if logger.restart_from_checkpoint:  # TODO  load ae+reg weights
            raise NotImplementedError()
        else:  # load VAE from a different path (must be given)
            pretrained_checkpoint = torch.load(config.model.pretrained_VAE_checkpoint, map_location=device)
            ae_model.load_checkpoint(pretrained_checkpoint)


    # ========== Scalars, metrics, images and audio to be tracked in Tensorboard ==========
    # Some of these metrics might be unused during pre-training
    # Special 'super-metrics', used by 1D scalars or metrics to retrieve stored data. Not directly logged
    super_metrics = {'LatentMetric/Train': LatentMetric(config.model.dim_z, dataloaders_nb_items['train']),
                     'LatentMetric/Valid': LatentMetric(config.model.dim_z, dataloaders_nb_items['validation']),
                     'RegOutValues/Train': VectorMetric(dataloaders_nb_items['train']),
                     'RegOutValues/Valid': VectorMetric(dataloaders_nb_items['validation'])}
    # 1D scalars with a .get() method. All of these will be automatically added to Tensorboard
    scalars = {  # Reconstruction loss (variable scale) + monitoring metrics comparable across all models
               'ReconsLoss/Backprop/Train': EpochMetric(), 'ReconsLoss/Backprop/Valid': EpochMetric(),
               'ReconsLoss/MSE/Train': EpochMetric(), 'ReconsLoss/MSE/Valid': EpochMetric(),
               # 'ReconsLoss/SC/Train': EpochMetric(), 'ReconsLoss/SC/Valid': EpochMetric(),  # TODO
               # Controls losses used for backprop + monitoring metrics (quantized numerical loss, categorical accuracy)
               'Controls/BackpropLoss/Train': EpochMetric(), 'Controls/BackpropLoss/Valid': EpochMetric(),
               'Controls/RegulLoss/Train': EpochMetric(), 'Controls/RegulLoss/Valid': EpochMetric(),
               'Controls/QLoss/Train': EpochMetric(), 'Controls/QLoss/Valid': EpochMetric(),
               'Controls/Accuracy/Train': EpochMetric(), 'Controls/Accuracy/Valid': EpochMetric(),
               # Latent-space and VAE losses
               'Latent/BackpropLoss/Train': EpochMetric(), 'Latent/BackpropLoss/Valid': EpochMetric(),
               'Latent/MMD/Train': EpochMetric(), 'Latent/MMD/Valid': EpochMetric(),
               'VAELoss/Train': SimpleMetric(), 'VAELoss/Valid': SimpleMetric(),
               # 'LatCorr/z0/Train': LatentCorrMetric(super_metrics['LatentMetric/Train'], 'z0'),  # Disabled
               # 'LatCorr/z0/Valid': LatentCorrMetric(super_metrics['LatentMetric/Valid'], 'z0'),  # (unused and
               # 'LatCorr/zK/Train': LatentCorrMetric(super_metrics['LatentMetric/Train'], 'zK'),  # very expensive)
               # 'LatCorr/zK/Valid': LatentCorrMetric(super_metrics['LatentMetric/Valid'], 'zK'),
               # Other misc. metrics
               'Sched/LRwarmup': LinearDynamicParam(
                   config.train.lr_warmup_start_factor, 1.0,
                   end_epoch=config.train.lr_warmup_epochs, current_epoch=config.train.start_epoch),
               'Sched/AttGamma': LinearDynamicParam(
                   0.0, config.model.attention_gamma,
                   end_epoch=(config.train.attention_gamma_warmup_period if pretrain_vae else 0)),
               'Sched/Controls/LR': SimpleMetric(config.train.initial_learning_rate['reg']),
               'Sched/VAE/LR': SimpleMetric(config.train.initial_learning_rate['ae']),
               'Sched/VAE/beta': LinearDynamicParam(config.train.beta_start_value, config.train.beta,
                                                    end_epoch=config.train.beta_warmup_epochs,
                                                    current_epoch=config.train.start_epoch) }
    # Validation metrics have a '_' suffix to be different from scalars (tensorboard mixes them)
    metrics = {'ReconsLoss/MSE/Valid_': logs.metrics.BufferedMetric(),
               'Latent/MMD/Valid_': logs.metrics.BufferedMetric(),
               'Latent/MaxAbsVal/Valid_': logs.metrics.BufferedMetric(),
               # 'LatCorr/z0/Valid_': logs.metrics.BufferedMetric(),
               # 'LatCorr/zK/Valid_': logs.metrics.BufferedMetric(),
               'Controls/QLoss/Valid_': logs.metrics.BufferedMetric(),
               'Controls/Accuracy/Valid_': logs.metrics.BufferedMetric(),
               'epochs': config.train.start_epoch}
    logger.tensorboard.init_hparams_and_metrics(metrics)  # hparams added knowing config.*


    # ========== PyTorch Profiling (optional) ==========
    optional_profiler = utils.profile.OptionalProfiler(config.train, logger.tensorboard_run_dir)


    # ========== Model training epochs ==========
    for epoch in range(config.train.start_epoch, config.train.n_epochs):
        # = = = = = Re-init of epoch metrics and useful scalars (warmup ramps, ...) = = = = =
        for _, s in super_metrics.items():
            s.on_new_epoch()
        for _, s in scalars.items():
            s.on_new_epoch()
        should_plot = (epoch % config.train.plot_period == 0)

        # = = = = = LR warmup (bypasses the scheduler during first epochs) = = = = =
        if epoch <= config.train.lr_warmup_epochs:
            ae_model.learning_rate = scalars['Sched/LRwarmup'].get(epoch) * config.train.initial_learning_rate['ae']
            reg_model.learning_rate = scalars['Sched/LRwarmup'].get(epoch) * config.train.initial_learning_rate['reg']
        ae_model.encoder.set_attention_gamma(scalars['Sched/AttGamma'].get(epoch))
        ae_model.decoder.set_attention_gamma(scalars['Sched/AttGamma'].get(epoch))

        # = = = = = Train all mini-batches (optional profiling) = = = = =
        # when profiling is disabled: true no-op context manager, and prof is None
        with optional_profiler.get_prof(epoch) as prof:
            ae_model_parallel.train()
            reg_model_parallel.train()
            dataloader_iter = iter(dataloader['train'])
            for i in range(len(dataloader['train'])):
                sample = next(dataloader_iter)
                x_in, v_in, sample_info = sample[0].to(device), sample[1].to(device), sample[2].to(device)
                reg_model.precompute_u_in_permutations(v_in)
                ae_model.optimizer.zero_grad()
                reg_model.optimizer.zero_grad()
                ae_out = ae_model_parallel(x_in, sample_info)  # Spectral VAE - tuple output
                z_0_mu_logvar, z_0_sampled, z_K_sampled, log_abs_det_jac, x_out = ae_out
                v_out = reg_model_parallel(z_K_sampled)  # returns a dummy zero during pre-train
                reg_model.precompute_u_out_with_symmetries(v_out)
                super_metrics['LatentMetric/Train'].append(z_0_mu_logvar, z_0_sampled, z_K_sampled)
                # Losses
                recons_loss = ae_model.reconstruction_loss(x_out, x_in)
                scalars['ReconsLoss/Backprop/Train'].append(recons_loss)
                # Latent loss computed on 1 GPU using the ae_model itself (not its parallelized version)
                lat_loss = ae_model.latent_loss(z_0_mu_logvar, z_0_sampled, z_K_sampled, log_abs_det_jac)
                scalars['Latent/BackpropLoss/Train'].append(lat_loss)
                lat_loss *= scalars['Sched/VAE/beta'].get(epoch)
                with torch.no_grad():  # Monitoring-only losses
                    scalars['ReconsLoss/MSE/Train'].append(ae_model.monitoring_reconstruction_loss(x_out, x_in))
                    scalars['Latent/MMD/Train'].append(ae_model.mmd(z_K_sampled))  # TODO don't compute twice
                    if not pretrain_vae:
                        accuracy, numerical_error = reg_model.eval_criterion_values
                        scalars['Controls/QLoss/Train'].append(numerical_error)
                        scalars['Controls/Accuracy/Train'].append(accuracy)
                        super_metrics['RegOutValues/Train'].append(v_out)
                extra_lat_reg_loss = ae_model.additional_latent_regularization_loss(z_0_mu_logvar)  # Might be 0.0
                extra_lat_reg_loss *= scalars['Sched/VAE/beta'].get(epoch)
                if not pretrain_vae:
                    if config.model.forward_controls_loss:
                        cont_loss = reg_model.backprop_loss_value
                    else:
                        cont_loss = reg_model.backprop_criterion(z_0_mu_logvar, v_in)  # FIXME
                    cont_loss *= config.train.params_loss_compensation_factor
                    scalars['Controls/BackpropLoss/Train'].append(cont_loss)
                    cont_reg_loss = reg_model.regularization_loss(v_out, v_in)
                    scalars['Controls/RegulLoss/Train'].append(cont_reg_loss)
                else:
                    cont_loss, cont_reg_loss = torch.zeros((1,), device=device), torch.zeros((1,), device=device)
                utils.exception.check_nan_values(
                    epoch, recons_loss, lat_loss, extra_lat_reg_loss, cont_loss, cont_reg_loss)
                # Backprop and optimizers' step (before schedulers' step)
                (recons_loss + lat_loss + extra_lat_reg_loss + cont_loss + cont_reg_loss).backward()
                ae_model.optimizer.step(), reg_model.optimizer.step()
                # End of mini-batch (step)
                logger.on_minibatch_finished(i)
                if prof is not None:
                    prof.step()
        scalars['VAELoss/Train'].set(scalars['ReconsLoss/Backprop/Train'].get()
                                     + scalars['Latent/BackpropLoss/Train'].get())

        # = = = = = Evaluation on validation dataset (no profiling) = = = = =
        with torch.no_grad():
            ae_model_parallel.eval()  # BN stops running estimates
            reg_model_parallel.eval()
            v_out_backup = torch.Tensor().to(device=recons_loss.device)  # Params inference error (Tensorboard plot)
            v_in_backup = torch.Tensor().to(device=recons_loss.device)
            i_to_plot = np.random.default_rng(seed=epoch).integers(0, len(dataloader['validation'])-1)
            for i, sample in enumerate(dataloader['validation']):
                x_in, v_in, sample_info = sample[0].to(device), sample[1].to(device), sample[2].to(device)
                reg_model.precompute_u_in_permutations(v_in)
                ae_out = ae_model_parallel(x_in, sample_info)  # Spectral VAE - tuple output
                z_0_mu_logvar, z_0_sampled, z_K_sampled, log_abs_det_jac, x_out = ae_out
                v_out = reg_model_parallel(z_K_sampled)
                reg_model.precompute_u_out_with_symmetries(v_out)
                super_metrics['LatentMetric/Valid'].append(z_0_mu_logvar, z_0_sampled, z_K_sampled)
                recons_loss = ae_model.reconstruction_loss(x_out, x_in)
                scalars['ReconsLoss/Backprop/Valid'].append(recons_loss)
                lat_loss = ae_model.latent_loss(z_0_mu_logvar, z_0_sampled, z_K_sampled, log_abs_det_jac)
                scalars['Latent/BackpropLoss/Valid'].append(lat_loss)
                # lat_loss *= scalars['Sched/beta'].get(epoch)  # Warmup factor: useless for monitoring
                # Monitoring losses
                scalars['ReconsLoss/MSE/Valid'].append(ae_model.monitoring_reconstruction_loss(x_out, x_in))
                scalars['Latent/MMD/Valid'].append(ae_model.mmd(z_K_sampled))
                if not pretrain_vae:
                    accuracy, numerical_error = reg_model.eval_criterion_values
                    scalars['Controls/QLoss/Valid'].append(numerical_error)
                    scalars['Controls/Accuracy/Valid'].append(accuracy)
                    super_metrics['RegOutValues/Valid'].append(v_out)
                    if config.model.forward_controls_loss:
                        cont_loss = reg_model.backprop_loss_value
                    else:
                        cont_loss = reg_model.backprop_criterion(z_0_mu_logvar, v_in)
                    cont_loss *= config.train.params_loss_compensation_factor
                    scalars['Controls/BackpropLoss/Valid'].append(cont_loss)
                    cont_reg_loss = reg_model.regularization_loss(v_out, v_in)
                    scalars['Controls/RegulLoss/Valid'].append(cont_reg_loss)
                # Validation plots
                if should_plot:
                    v_out_backup = torch.cat([v_out_backup, v_out])  # Full-batch error storage - will be used later
                    v_in_backup = torch.cat([v_in_backup, v_in])
                    if i == i_to_plot:  # random mini-batch plot (validation dataset is not randomized)
                        logger.plot_train_spectrograms(epoch, x_in, x_out, sample_info,
                                                       dataset if dataset is not None else validation_audio_dataset,
                                                       config.model, config.train)
                        # "CUDA illegal memory access (stacktrace might be incorrect)" - fixed by torch 1.10, CUDA 11.3
                        logger.plot_decoder_interpolation(epoch, ae_model, z_K_sampled, sample_info,
                                                          dataset if dataset is not None else validation_audio_dataset)
        metrics['Latent/MaxAbsVal/Valid_'].append(np.abs(super_metrics['LatentMetric/Valid'].get_z('zK')).max())
        scalars['VAELoss/Valid'].set(scalars['ReconsLoss/Backprop/Valid'].get()
                                     + scalars['Latent/BackpropLoss/Valid'].get())
        # Dynamic LR scheduling depends on validation performance
        # Summed losses for plateau-detection are chosen in config.py
        if pretrain_vae or (not pretrain_vae and config.train.enable_ae_scheduler_after_pretrain):
            if config.train.scheduler_name == 'ReduceLROnPlateau':
                ae_model.scheduler.step(sum([scalars['{}/Valid'.format(loss_name)].get()
                                             for loss_name in config.train.scheduler_losses['ae']]))
            else:  # deterministic scheduler
                ae_model.scheduler.step()
        scalars['Sched/VAE/LR'].set(ae_model.learning_rate)
        if not pretrain_vae:
            if config.train.scheduler_name == 'ReduceLROnPlateau':
                reg_model.scheduler.step(sum([scalars['{}/Valid'.format(loss_name)].get()
                                              for loss_name in config.train.scheduler_losses['reg']]))
            else:  # deterministic scheduler
                reg_model.scheduler.step()
            scalars['Sched/Controls/LR'].set(reg_model.learning_rate)
        # Possible early stop if reg model is not learning anything anymore
        early_stop = (reg_model.learning_rate < config.train.early_stop_lr_threshold['reg'])

        # = = = = = Epoch logs (scalars/sounds/images + updated metrics) = = = = =
        for k, s in scalars.items():  # All available scalars are written to tensorboard
            try:
                # don't need to log everything for every train procedure (during pre-train, if no flow, etc...)
                if (k.startswith('Controls') and pretrain_vae) or (k.startswith('Sched/Controls') and pretrain_vae)\
                        or (k.startswith('LatCorr/zK') and config.model.latent_flow_arch is None):
                    pass
                else:
                    logger.tensorboard.add_scalar(k, s.get(), epoch)  # .get might raise except if empty/unused scalar
            except ValueError:  # unused scalars with buffer (e.g. during pretrain) will raise that exception
                pass
        if should_plot or early_stop:
            logger.plot_stats_tensorboard__threaded(config.train, epoch, super_metrics, ae_model)  # non-blocking
            if v_in_backup.shape[0] > 0 and not pretrain_vae:  # u_error might be empty on early_stop
                fig, _ = utils.figures.plot_synth_preset_vst_error(
                    v_out_backup.detach().cpu(), v_in_backup.detach().cpu(), dataset.preset_indexes_helper)
                logger.tensorboard.add_figure('SynthControlsError', fig, epoch)
        metrics['epochs'] = epoch + 1
        metrics['ReconsLoss/MSE/Valid_'].append(scalars['ReconsLoss/MSE/Valid'].get())
        metrics['Latent/MMD/Valid_'].append(scalars['Latent/MMD/Valid'].get())
        # metrics['LatCorr/z0/Valid_'].append(scalars['LatCorr/z0/Valid'].get())
        if config.model.latent_flow_arch is not None:
            pass  # metrics['LatCorr/zK/Valid_'].append(scalars['LatCorr/zK/Valid'].get())
        if not pretrain_vae:
            metrics['Controls/QLoss/Valid_'].append(scalars['Controls/QLoss/Valid'].get())
            metrics['Controls/Accuracy/Valid_'].append(scalars['Controls/Accuracy/Valid'].get())
        logger.tensorboard.update_metrics(metrics)

        # = = = = = Model+optimizer(+scheduler) save - ready for next epoch = = = = =
        if (epoch > 0 and epoch % config.train.save_period == 0)\
                or (epoch == config.train.n_epochs-1) or early_stop:
            logger.save_checkpoint(epoch, ae_model, (None if pretrain_vae else reg_model))
        logger.on_epoch_finished(epoch)
        if early_stop:
            print("[train.py] Training stopped early (final loss plateau)")
            break


    # ========== Logger final stats ==========
    logger.on_training_finished()  # Might have to wait for threads


    # ========== "Manual GC" (to try to prevent random CUDA out-of-memory between enqueued runs ==========
    del reg_model_parallel, ae_model_parallel
    del extended_ae_model, ae_model
    del reg_model
    del v_in, v_out, v_in_backup, v_out_backup, x_in, x_out, sample_info
    del ae_out, z_0_sampled, z_K_sampled, z_0_mu_logvar
    del extra_lat_reg_loss, lat_loss
    del metrics, super_metrics
    del logger
    del dataloader, dataset
    del train_audio_dataset, validation_audio_dataset
    gc.collect()


if __name__ == "__main__":
    # Normal run, config.py only will be used to parametrize learning and models
    config.update_dynamic_config_params()  # Required before any actual train
    train_config()

