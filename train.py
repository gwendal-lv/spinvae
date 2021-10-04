"""
This script performs a single training run for the configuration described
in config.py, when running as __main__.

Its train_config(...) function can also be called from another script,
with small modifications to the config (enqueued train runs).

See train_queue.py for enqueued training runs
"""
import multiprocessing
from pathlib import Path
import contextlib
from typing import Optional, Dict, List

import numpy as np
import mkl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim
from torch.autograd import profiler

import config
import model.base
import model.loss
import model.build
import model.extendedAE
import model.flows
import logs.logger
import logs.metrics
from logs.metrics import SimpleMetric, EpochMetric, LatentMetric, LatentCorrMetric
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
        dataloader, dataloaders_nb_items = data.build.get_split_dataloaders(config.train, dataset)


    # ========== Logger init (required to load from checkpoint) and Config check ==========
    root_path = Path(__file__).resolve().parent
    logger = logs.logger.RunLogger(root_path, config.model, config.train)
    if logger.restart_from_checkpoint:
        model.build.check_configs_on_resume_from_checkpoint(config.model, config.train,
                                                            logger.get_previous_config_from_json())
    if pretrain_vae:
        if config.train.start_epoch > 0:  # Resume from checkpoint?
            raise NotImplementedError()  # TODO load the "pretrain" checkpoint"
            start_checkpoint = logs.logger.get_model_checkpoint(root_path, config.model, config.train.start_epoch - 1)
        else:
            start_checkpoint = None
    else:
        raise NotImplementedError()  # TODO  we should load the AE model from a different checkpoint


    # ========== Model definition (requires the full_dataset to be built) ==========
    # The extended_ae_model has all sub-models as attributes (even if some sub-models, e.g. synth controls regression,
    # are set to None during pre-training). Useful to change device or train/eval status of all models.
    if pretrain_vae:
        _, _, ae_model = model.build.build_ae_model(config.model, config.train)
        reg_model = model.base.DummyModel()
        extended_ae_model = model.extendedAE.ExtendedAE(ae_model, reg_model)
    else:
        _, _, ae_model, reg_model, extended_ae_model = model.build.build_extended_ae_model(config.model, config.train,
                                                                                           preset_indexes_helper)
    if start_checkpoint is not None:
        # FIXME models should load the checkpoint themselves
        ae_model.load_state_dict(start_checkpoint['ae_model_state_dict'])  # GPU tensor params
        if not pretrain_vae:
            raise NotImplementedError()
            # TODO separate ae and reg/extended ae models
            #    we should load the AE model from a different checkpoint ! and we need a new fct for that
    extended_ae_model.eval()
    # will write tensorboard graph and torchinfo txt summary. model must not be parallel
    logger.init_with_model(ae_model, config.model.input_tensor_size)  # main model: autoencoder
    if not isinstance(reg_model, model.base.DummyModel):
        logger.write_model_summary(reg_model, (config.train.minibatch_size, config.model.dim_z), "reg")  # Other model


    # ========== Training devices (GPU(s) only) ==========
    if config.train.verbosity >= 1:
        print("Intel MKL num threads = {}. PyTorch num threads = {}. CUDA devices count: {} GPU(s)."
              .format(mkl.get_max_threads(), torch.get_num_threads(), torch.cuda.device_count()))
    if torch.cuda.device_count() == 0:
        raise NotImplementedError()  # CPU training not available
    elif torch.cuda.device_count() == 1 or config.train.profiler_1_GPU:
        if config.train.profiler_1_GPU:
            print("Using 1/{} GPUs for code profiling".format(torch.cuda.device_count()))
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


    # ========== Losses (criterion functions) ==========
    # Training losses (for backprop) and Metrics (monitoring) losses and accuracies
    # Some losses are defined in the models themselves
    if config.train.normalize_losses:  # Reconstruction backprop loss
        reconstruction_criterion = nn.MSELoss(reduction='mean')
    else:
        reconstruction_criterion = model.loss.L2Loss()
    if not pretrain_vae:
        # Controls backprop loss
        if config.model.forward_controls_loss:  # usual straightforward loss - compares inference and target
            if config.train.params_cat_bceloss:
                assert (not config.model.params_reg_softmax)  # BCE loss requires no-softmax at reg model output
            controls_criterion = model.loss.SynthParamsLoss(preset_indexes_helper,
                                                            config.train.normalize_losses,
                                                            cat_bce=config.train.params_cat_bceloss,
                                                            cat_softmax=(not config.model.params_reg_softmax
                                                                         and not config.train.params_cat_bceloss),
                                                            cat_softmax_t=config.train.params_cat_softmax_temperature)
        else:
            raise ValueError("Backward-computed synth params regression loss: deprecated")
        # Monitoring losses always remain the same
        controls_num_eval_criterion = model.loss.QuantizedNumericalParamsLoss(preset_indexes_helper,
                                                                              numerical_loss=nn.MSELoss(
                                                                                  reduction='mean'))
        controls_accuracy_criterion = model.loss.CategoricalParamsAccuracy(preset_indexes_helper,
                                                                           reduce=True, percentage_output=True)
    else:
        controls_criterion, controls_num_eval_criterion, controls_accuracy_criterion = None, None, None
    # Stabilizing loss for flow-based latent space
    if config.train.latent_flow_input_regularization.lower() == 'dkl':
        # TODO check (model must use a latent flow)
        flow_input_dkl = model.loss.GaussianDkl(normalize=config.train.normalize_losses)


    # ========== Scalars, metrics, images and audio to be tracked in Tensorboard ==========
    # Some of these metrics might be unused during pre-training
    # Special 'super-metrics', used by 1D scalars or metrics to retrieve stored data. Not directly logged
    super_metrics = {'LatentMetric/Train': LatentMetric(config.model.dim_z, dataloaders_nb_items['train']),
                     'LatentMetric/Valid': LatentMetric(config.model.dim_z, dataloaders_nb_items['validation'])}
    # 1D scalars with a .get() method. All of these will be automatically added to Tensorboard
    scalars = {  # Reconstruction loss (variable scale) + monitoring metrics comparable across all models
               'ReconsLoss/Backprop/Train': EpochMetric(), 'ReconsLoss/Backprop/Valid': EpochMetric(),
               'ReconsLoss/MSE/Train': EpochMetric(), 'ReconsLoss/MSE/Valid': EpochMetric(),
                # TODO maybe don't add controls losses ?
               # 'ReconsLoss/SC/Train': EpochMetric(), 'ReconsLoss/SC/Valid': EpochMetric(),  # TODO
               # Controls losses used for backprop + monitoring metrics (quantized numerical loss, categorical accuracy)
               'Controls/BackpropLoss/Train': EpochMetric(), 'Controls/BackpropLoss/Valid': EpochMetric(),
               'Controls/QLoss/Train': EpochMetric(), 'Controls/QLoss/Valid': EpochMetric(),
               'Controls/Accuracy/Train': EpochMetric(), 'Controls/Accuracy/Valid': EpochMetric(),
               # Latent-space and VAE losses
               'LatLoss/Train': EpochMetric(), 'LatLoss/Valid': EpochMetric(),
               'VAELoss/Train': SimpleMetric(), 'VAELoss/Valid': SimpleMetric(),
               'LatCorr/z0/Train': LatentCorrMetric(super_metrics['LatentMetric/Train'], 'z0'),
               'LatCorr/z0/Valid': LatentCorrMetric(super_metrics['LatentMetric/Valid'], 'z0'),
               'LatCorr/zK/Train': LatentCorrMetric(super_metrics['LatentMetric/Train'], 'zK'),
               'LatCorr/zK/Valid': LatentCorrMetric(super_metrics['LatentMetric/Valid'], 'zK'),
               # Other misc. metrics
               'Sched/LRwarmup': LinearDynamicParam(config.train.lr_warmup_start_factor, 1.0,
                                                    end_epoch=config.train.lr_warmup_epochs,
                                                    current_epoch=config.train.start_epoch),
               'Sched/Controls/LR': SimpleMetric(config.train.initial_learning_rate['reg']),
               'Sched/VAE/LR': SimpleMetric(config.train.initial_learning_rate['ae']),
               'Sched/VAE/beta': LinearDynamicParam(config.train.beta_start_value, config.train.beta,
                                                    end_epoch=config.train.beta_warmup_epochs,
                                                    current_epoch=config.train.start_epoch) }
    # Validation metrics have a '_' suffix to be different from scalars (tensorboard mixes them)
    metrics = {'ReconsLoss/MSE/Valid_': logs.metrics.BufferedMetric(),
               'LatLoss/Valid_': logs.metrics.BufferedMetric(),
               'LatCorr/z0/Valid_': logs.metrics.BufferedMetric(),
               'LatCorr/zK/Valid_': logs.metrics.BufferedMetric(),
                # TODO maybe don't add controls losses ?
               'Controls/QLoss/Valid_': logs.metrics.BufferedMetric(),
               'Controls/Accuracy/Valid_': logs.metrics.BufferedMetric(),
               'epochs': config.train.start_epoch}
    logger.tensorboard.init_hparams_and_metrics(metrics)  # hparams added knowing config.*


    # ========== Optimizer and Scheduler ==========
    ae_model.init_optimizer_and_scheduler()
    reg_model.init_optimizer_and_scheduler()
    if start_checkpoint is not None:
        raise NotImplementedError()  # TODO handle pretrain/not cases (we won't load stats dict from the same checkpoint)
        optimizer.load_state_dict(start_checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(start_checkpoint['scheduler_state_dict'])


    # ========== PyTorch Profiling (optional) ==========
    is_profiled = config.train.profiler_args['enabled']
    extended_ae_model.is_profiled = is_profiled


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

        # = = = = = Train all mini-batches (optional profiling) = = = = =
        # when profiling is disabled: true no-op context manager, and prof is None
        with utils.profile.get_optional_profiler(config.train.profiler_args) as prof:
            ae_model_parallel.train()
            reg_model_parallel.train()
            dataloader_iter = iter(dataloader['train'])
            for i in range(len(dataloader['train'])):
                with profiler.record_function("DATA_LOAD") if is_profiled else contextlib.nullcontext():
                    sample = next(dataloader_iter)
                    x_in, v_in, sample_info = sample[0].to(device), sample[1].to(device), sample[2].to(device)
                with profiler.record_function("FORWARD") if is_profiled else contextlib.nullcontext():
                    ae_model.optimizer.zero_grad()
                    reg_model.optimizer.zero_grad()
                    ae_out = ae_model_parallel(x_in, sample_info)  # Spectral VAE - tuple output
                    z_0_mu_logvar, z_0_sampled, z_K_sampled, log_abs_det_jac, x_out = ae_out
                    v_out = reg_model_parallel(z_K_sampled)  # returns a dummy zero during pre-train
                with profiler.record_function("LOSSES") if is_profiled else contextlib.nullcontext():
                    super_metrics['LatentMetric/Train'].append(z_0_mu_logvar, z_0_sampled, z_K_sampled)
                    recons_loss = reconstruction_criterion(x_out, x_in)
                    scalars['ReconsLoss/Backprop/Train'].append(recons_loss)
                    # Latent loss computed on 1 GPU using the ae_model itself (not its parallelized version)
                    lat_loss = extended_ae_model.latent_loss(z_0_mu_logvar, z_0_sampled, z_K_sampled, log_abs_det_jac)
                    scalars['LatLoss/Train'].append(lat_loss)
                    lat_loss *= scalars['Sched/VAE/beta'].get(epoch)
                    # Monitoring losses
                    with torch.no_grad():
                        scalars['ReconsLoss/MSE/Train'].append(recons_loss)
                        if not pretrain_vae:  # TODO refactor....
                            scalars['Controls/QLoss/Train'].append(controls_num_eval_criterion(v_out, v_in))
                            scalars['Controls/Accuracy/Train'].append(controls_accuracy_criterion(v_out, v_in))
                    # Flow training stabilization loss?
                    flow_input_loss = torch.tensor([0.0], device=lat_loss.device)
                    if extended_ae_model.is_flow_based_latent_space and\
                            (config.train.latent_flow_input_regularization.lower() == 'dkl'):
                        flow_input_loss = config.train.latent_flow_input_regul_weight * config.train.beta *\
                                          flow_input_dkl(z_0_mu_logvar[:, 0, :], z_0_mu_logvar[:, 1, :])
                    if not pretrain_vae:
                        if config.model.forward_controls_loss:  # unused params might be modified by this criterion
                            cont_loss = controls_criterion(v_out, v_in)
                        else:
                            cont_loss = controls_criterion(z_0_mu_logvar, v_in)
                        scalars['Controls/BackpropLoss/Train'].append(cont_loss)
                    else:
                        cont_loss = torch.zeros((1,), device=device)
                    utils.exception.check_nan_values(epoch, recons_loss, lat_loss, flow_input_loss, cont_loss)
                with profiler.record_function("BACKPROP") if is_profiled else contextlib.nullcontext():
                    (recons_loss + lat_loss + flow_input_loss + cont_loss).backward()  # Actual backpropagation is here
                with profiler.record_function("OPTIM_STEP") if is_profiled else contextlib.nullcontext():
                    ae_model.optimizer.step()  # Internal params. update; before scheduler step
                    reg_model.optimizer.step()
                logger.on_minibatch_finished(i)
                # For full-trace profiling: we need to stop after a few mini-batches
                if config.train.profiler_full_trace and i == 1:
                    break
        if prof is not None:
            logger.save_profiler_results(prof, config.train.profiler_full_trace)
        if config.train.profiler_full_trace:
            break  # Forced training stop
        scalars['VAELoss/Train'] = SimpleMetric(scalars['ReconsLoss/Backprop/Train'].get()
                                                + scalars['LatLoss/Train'].get())

        # = = = = = Evaluation on validation dataset (no profiling) = = = = =
        with torch.no_grad():
            ae_model_parallel.eval()  # BN stops running estimates
            if reg_model_parallel is not None:
                reg_model_parallel.eval()
            v_error = torch.Tensor().to(device=recons_loss.device)  # Params inference error (Tensorboard plot)
            for i, sample in enumerate(dataloader['validation']):
                x_in, v_in, sample_info = sample[0].to(device), sample[1].to(device), sample[2].to(device)
                ae_out = ae_model_parallel(x_in, sample_info)  # Spectral VAE - tuple output
                z_0_mu_logvar, z_0_sampled, z_K_sampled, log_abs_det_jac, x_out = ae_out
                v_out = reg_model_parallel(z_K_sampled)
                super_metrics['LatentMetric/Valid'].append(z_0_mu_logvar, z_0_sampled, z_K_sampled)
                recons_loss = reconstruction_criterion(x_out, x_in)
                scalars['ReconsLoss/Backprop/Valid'].append(recons_loss)
                lat_loss = extended_ae_model.latent_loss(z_0_mu_logvar, z_0_sampled, z_K_sampled, log_abs_det_jac)
                scalars['LatLoss/Valid'].append(lat_loss)
                # lat_loss *= scalars['Sched/beta'].get(epoch)  # Warmup factor: useless for monitoring
                # Monitoring losses
                scalars['ReconsLoss/MSE/Valid'].append(recons_loss)
                if not pretrain_vae:
                    scalars['Controls/QLoss/Valid'].append(controls_num_eval_criterion(v_out, v_in))
                    scalars['Controls/Accuracy/Valid'].append(controls_accuracy_criterion(v_out, v_in))
                    if config.model.forward_controls_loss:  # unused params might be modified by this criterion
                        cont_loss = controls_criterion(v_out, v_in)
                    else:
                        cont_loss = controls_criterion(z_0_mu_logvar, v_in)
                    scalars['Controls/BackpropLoss/Valid'].append(cont_loss)
                # Validation plots
                if should_plot:
                    v_error = torch.cat([v_error, v_out - v_in])  # Full-batch error storage - will be used later
                    if i == 0:  # tensorboard samples for minibatch 'eval' [0] only
                        fig, _ = utils.figures.\
                            plot_train_spectrograms(x_in, x_out, sample_info,
                                                    dataset if dataset is not None else validation_audio_dataset,
                                                    config.model, config.train)
                        logger.tensorboard.add_figure('Spectrogram', fig, epoch, close=True)

        scalars['VAELoss/Valid'] = SimpleMetric(scalars['ReconsLoss/Backprop/Valid'].get()
                                                + scalars['LatLoss/Valid'].get())
        # Dynamic LR scheduling depends on validation performance
        # Summed losses for plateau-detection are chosen in config.py
        ae_model.scheduler.step(sum([scalars['{}/Valid'.format(loss_name)].get()
                                     for loss_name in config.train.scheduler_losses['ae']]))
        scalars['Sched/VAE/LR'] = logs.metrics.SimpleMetric(ae_model.learning_rate)
        if not pretrain_vae:
            reg_model.scheduler.step(sum([scalars['{}/Valid'.format(loss_name)].get()
                                          for loss_name in config.train.scheduler_losses['reg']]))
            scalars['Sched/Controls/LR'] = logs.metrics.SimpleMetric(reg_model.learning_rate)
        # Possible early stop is reg model is not learning anything anymore
        early_stop = (reg_model.learning_rate < config.train.early_stop_lr_threshold['reg'])

        # = = = = = Epoch logs (scalars/sounds/images + updated metrics) = = = = =
        for k, s in scalars.items():  # All available scalars are written to tensorboard
            try:
                logger.tensorboard.add_scalar(k, s.get(), epoch)  # .get might raise except if empty/unused scalar
            except ValueError:  # unused scalars with buffer (e.g. during pretrain) will raise that exception
                pass
        if should_plot or early_stop:
            logger.plot_latent_stats_tensorboard(epoch, super_metrics)  # separate thread and process (-9s / plot epoch)
            if v_error.shape[0] > 0 and not pretrain_vae:  # u_error might be empty on early_stop
                fig, _ = utils.figures.plot_synth_preset_error(v_error.detach().cpu(), dataset.preset_indexes_helper)
                logger.tensorboard.add_figure('SynthControlsError', fig, epoch)
        metrics['epochs'] = epoch + 1
        metrics['ReconsLoss/MSE/Valid_'].append(scalars['ReconsLoss/MSE/Valid'].get())
        metrics['LatLoss/Valid_'].append(scalars['LatLoss/Valid'].get())
        metrics['LatCorr/z0/Valid_'].append(scalars['LatCorr/z0/Valid'].get())
        metrics['LatCorr/zK/Valid_'].append(scalars['LatCorr/zK/Valid'].get())
        if not pretrain_vae:  # TODO handle properly
            metrics['Controls/QLoss/Valid_'].append(scalars['Controls/QLoss/Valid'].get())
            metrics['Controls/Accuracy/Valid_'].append(scalars['Controls/Accuracy/Valid'].get())
        logger.tensorboard.update_metrics(metrics)


        # = = = = = Model+optimizer(+scheduler) save - ready for next epoch = = = = =
        # TODO properly save
        if (epoch > 0 and epoch % config.train.save_period == 0)\
                or (epoch == config.train.n_epochs-1) or early_stop:
            pass
            #logger.save_checkpoint(epoch, extended_ae_model, optimizer, scheduler)  # FIXME
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
    del controls_criterion, controls_num_eval_criterion, controls_accuracy_criterion, reconstruction_criterion
    del logger
    del dataloader, dataset
    del train_audio_dataset, validation_audio_dataset


if __name__ == "__main__":
    # Normal run, config.py only will be used to parametrize learning and models
    train_config()

