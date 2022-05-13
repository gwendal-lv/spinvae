import copy
import multiprocessing
import threading
import os
import sys
import time
import shutil
import json
import datetime
import pathlib
import warnings
from typing import List, Optional

import numpy as np
import torch
import torchinfo

import humanize

from data.abstractbasedataset import AudioDataset
import model.base
import model.VAE
import utils
import utils.figures
import utils.stat
from .tbwriter import TensorboardSummaryWriter  # Custom modified summary writer
import logs.logger_mp

_erase_security_time_s = 5.0


def get_model_run_directory(root_path, model_config):
    """ Returns the directory where saved models and config.json are stored, for a particular run.
    Does not check whether the directory exists or not (it must have been created by the RunLogger) """
    return root_path.joinpath(model_config.logs_root_dir)\
        .joinpath(model_config.name).joinpath(model_config.run_name)


def get_model_checkpoints_dir(root_path: pathlib.Path, model_config):
    return root_path.joinpath(model_config.logs_root_dir).joinpath(model_config.name)\
        .joinpath(model_config.run_name).joinpath('checkpoints')


def get_model_checkpoint_from_dir(checkpoints_dir: pathlib.Path, epoch, device=None):
    checkpoint_path = checkpoints_dir.joinpath('{:05d}.tar'.format(epoch))
    try:
        if device is None:
            checkpoint = torch.load(checkpoint_path)  # Load on original device
        else:
            checkpoint = torch.load(checkpoint_path, map_location=device)  # e.g. train on GPU, load on CPU
    except (OSError, IOError) as e:
        available_checkpoints = "Available checkpoints: {}".format([f.name for f in checkpoints_dir.glob('*.tar')])
        print(available_checkpoints)
        raise ValueError("Cannot load checkpoint for epoch {}: {}".format(epoch, e))
    return checkpoint


def get_model_checkpoint(root_path: pathlib.Path, model_config, epoch, device=None):
    """ Returns the path to a .tar saved checkpoint, or prints all available checkpoints and raises an exception
    if the required epoch has no saved .tar checkpoint. """
    checkpoints_dir = get_model_checkpoints_dir(root_path, model_config)
    return get_model_checkpoint_from_dir(checkpoints_dir, epoch, device)


def get_model_last_checkpoint(root_path: pathlib.Path, model_config, verbose=True, device=None):
    checkpoints_dir = root_path.joinpath(model_config.logs_root_dir).joinpath(model_config.name)\
        .joinpath(model_config.run_name).joinpath('checkpoints')
    available_epochs = [int(f.stem) for f in checkpoints_dir.glob('*.tar')]
    assert len(available_epochs) > 0  # At least 1 checkpoint should be available
    if verbose:
        print("Loading epoch {} from {}".format(max(available_epochs), checkpoints_dir))
    return get_model_checkpoint(root_path, model_config, max(available_epochs), device)





def get_tensorboard_run_directory(root_path, model_config):
    """ Returns the directory where Tensorboard model metrics are stored, for a particular run. """
    # pb s'il y en a plusieurs ? (semble résolu avec override de add_hparam PyTorch)
    return root_path.joinpath(model_config.logs_root_dir).joinpath('runs')\
        .joinpath(model_config.name).joinpath(model_config.run_name)


def erase_run_data(root_path, model_config):
    """ Erases all previous data (Tensorboard, config, saved models)
    for a particular run of the model. """
    if _erase_security_time_s > 0.1:
        print("[RunLogger] *** WARNING *** '{}' run for model '{}' will be erased in {} seconds. "
              "Stop this program to cancel ***"
              .format(model_config.run_name, model_config.name, _erase_security_time_s))
        time.sleep(_erase_security_time_s)
    else:
        print("[RunLogger] '{}' run for model '{}' will be erased.".format(model_config.run_name, model_config.name))
    shutil.rmtree(get_model_run_directory(root_path, model_config))  # config and saved models
    shutil.rmtree(get_tensorboard_run_directory(root_path, model_config))  # tensorboard


class RunLogger:
    """ Class for saving interesting data during a training run:
     - graphs, losses, metrics, and some results to Tensorboard
     - config.py as a json file
     - trained models

     See ../README.md to get more info on storage location.
     """
    def __init__(self, root_path, model_config, train_config, minibatches_count=0,
                 use_multiprocessing=True):
        """

        :param root_path: pathlib.Path of the project's root folder
        :param model_config: from config.py
        :param train_config: from config.py
        :param minibatches_count: Length of the 'train' dataloader
        """
        # Configs are stored but not modified by this class
        self.model_config = model_config
        self.train_config = train_config
        self.verbosity = train_config.verbosity
        global _erase_security_time_s  # Very dirty.... but quick
        _erase_security_time_s = train_config.init_security_pause
        assert train_config.start_epoch >= 0  # Cannot start from a negative epoch
        self.restart_from_checkpoint = (train_config.start_epoch > 0)
        # - - - - - Directories creation (if not exists) for model - - - - -
        self.log_dir = root_path.joinpath(model_config.logs_root_dir).joinpath(model_config.name)
        self._make_dirs_if_dont_exist(self.log_dir)
        self.tensorboard_model_dir = root_path.joinpath(model_config.logs_root_dir)\
            .joinpath('runs').joinpath(model_config.name)
        self._make_dirs_if_dont_exist(self.tensorboard_model_dir)
        # - - - - - Run directories and data management - - - - -
        if self.train_config.verbosity >= 1:
            print("[RunLogger] Starting logging into '{}'".format(self.log_dir))
        self.run_dir = self.log_dir.joinpath(model_config.run_name)  # This is the run's reference folder
        self.checkpoints_dir = self.run_dir.joinpath('checkpoints')
        self.tensorboard_run_dir = self.tensorboard_model_dir.joinpath(model_config.run_name)
        # Check: does the run folder already exist?
        if not os.path.exists(self.run_dir):
            if train_config.start_epoch != 0:
                raise RuntimeError("config.py error: this new run must start from epoch 0")
            self._make_model_run_dirs()
            if self.train_config.verbosity >= 1:
                print("[RunLogger] Created '{}' directory to store config and models.".format(self.run_dir))
        # If run folder already exists
        else:
            if self.restart_from_checkpoint:
                if self.verbosity >= 1:
                    print("[RunLogger] Will load saved checkpoint (previous epoch: {})"
                          .format(self.train_config.start_epoch - 1))
            else:  # Start a new fresh training
                if not model_config.allow_erase_run:
                    raise RuntimeError("Config does not allow to erase the '{}' run for model '{}'"
                                       .format(model_config.run_name, model_config.name))
                else:
                    erase_run_data(root_path, model_config)  # module function
                    self._make_model_run_dirs()
        # - - - - - Epochs, Batches, ... - - - - -
        self.minibatches_count = minibatches_count
        self.minibatch_duration_running_avg = 0.0
        self.minibatch_duration_avg_coeff = 0.05  # auto-regressive running average coefficient
        self.last_minibatch_start_datetime = datetime.datetime.now()
        self.epoch_start_datetimes = [datetime.datetime.now()]  # This value can be erased in init_with_model
        # - - - - - Tensorboard - - - - -
        self.tensorboard = TensorboardSummaryWriter(log_dir=self.tensorboard_run_dir, flush_secs=5,
                                                    model_config=model_config, train_config=train_config)
        # - - - - - Multi-processed plotting (plot time: approx. 20s / plotted epoch) - - - - -
        # Use multiprocessing if required by args, and if PyCharm debugger not detected
        self.use_multiprocessing = use_multiprocessing and (not (sys.gettrace() is not None))
        # Processes will be started and joined from those threads
        self.figures_threads = [None, None]  # type: List[Optional[threading.Thread]]

    @staticmethod
    def _make_dirs_if_dont_exist(dir_path):
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

    def _make_model_run_dirs(self):
        """ Creates (no check) the directories for storing config and saved models. """
        os.makedirs(self.run_dir)
        os.makedirs(self.checkpoints_dir)

    def init_with_model(self, main_model, input_tensor_size, write_graph=True):
        """ Finishes to initialize this logger given the fully-build model. This function must be called
         after all checks (configuration consistency, etc...) have been performed, because it overwrites files. """
        # Write config file on startup only - any previous config file will be erased
        # New/Saved configs compatibility must have been checked before calling this function
        config_dict = {'model': self.model_config.__dict__, 'train': self.train_config.__dict__}
        with open(self.run_dir.joinpath('config.json'), 'w') as f:
            json.dump(config_dict, f)
        if not self.restart_from_checkpoint:  # Graphs written at epoch 0 only
            self.write_model_summary(main_model, input_tensor_size, 'VAE')
            if write_graph:
                self.tensorboard.add_graph(main_model, torch.zeros(input_tensor_size))
        self.epoch_start_datetimes = [datetime.datetime.now()]

    def write_model_summary(self, model, input_tensor_size, model_name):
        if not self.restart_from_checkpoint:  # Graphs written at epoch 0 only
            description = torchinfo.summary(model, input_size=input_tensor_size, depth=6,
                                            device=torch.device('cpu'), verbose=0)
            with open(self.run_dir.joinpath('torchinfo_summary_{}.txt'.format(model_name)), 'w') as f:
                f.write(description.__str__())

    def get_previous_config_from_json(self):
        with open(self.run_dir.joinpath('config.json'), 'r') as f:
            full_config = json.load(f)
        return full_config

    def on_minibatch_finished(self, minibatch_idx):
        # TODO time stats - running average
        minibatch_end_time = datetime.datetime.now()
        delta_t = (minibatch_end_time - self.last_minibatch_start_datetime).total_seconds()
        self.minibatch_duration_running_avg *= (1.0 - self.minibatch_duration_avg_coeff)
        self.minibatch_duration_running_avg += self.minibatch_duration_avg_coeff * delta_t
        if self.verbosity >= 3:
            print("epoch {} batch {} delta t = {}ms" .format(len(self.epoch_start_datetimes)-1, minibatch_idx,
                                                             int(1000.0 * self.minibatch_duration_running_avg)))
        self.last_minibatch_start_datetime = minibatch_end_time

    def save_checkpoint(self, epoch, ae_model: model.base.TrainableModel,
                        reg_model: Optional[model.base.TrainableModel] = None):
        # These keys will be re-used by model/base.py, TrainableModel::load_checkpoint(...)
        checkpoint_dict = {'epoch': epoch,
                           'ae': {'model_state_dict': ae_model.state_dict(),
                                  'optimizer_state_dict': ae_model.optimizer.state_dict(),  # TODO maybe optional?
                                  'scheduler_state_dict': ae_model.scheduler.state_dict()}}
        if reg_model is not None:
            checkpoint_dict['reg'] = {'model_state_dict': reg_model.state_dict(),
                                      'optimizer_state_dict': reg_model.optimizer.state_dict(),
                                      'scheduler_state_dict': reg_model.scheduler.state_dict()}
        checkpoint_path = self.checkpoints_dir.joinpath('{:05d}.tar'.format(epoch))
        torch.save(checkpoint_dict, checkpoint_path)
        if self.verbosity >= 1:
            print("[RunLogger] Saved epoch {} checkpoint (models, optimizers, schedulers) to {}"
                  .format(epoch, checkpoint_path))

    def on_epoch_finished(self, epoch):
        self.epoch_start_datetimes.append(datetime.datetime.now())
        epoch_duration = self.epoch_start_datetimes[-1] - self.epoch_start_datetimes[-2]
        avg_duration_s = np.asarray([(self.epoch_start_datetimes[i+1] - self.epoch_start_datetimes[i]).total_seconds()
                                     for i in range(len(self.epoch_start_datetimes) - 1)])
        avg_duration_s = avg_duration_s.mean()
        run_total_epochs = self.train_config.n_epochs - self.train_config.start_epoch
        remaining_datetime = avg_duration_s * (run_total_epochs - (epoch-self.train_config.start_epoch) - 1)
        remaining_datetime = datetime.timedelta(seconds=int(remaining_datetime))
        if self.verbosity >= 1:
            print("End of epoch {} ({}/{}). Duration={:.1f}s, avg={:.1f}s. Estimated remaining time: {} ({})"
                  .format(epoch, epoch-self.train_config.start_epoch+1, run_total_epochs,
                          epoch_duration.total_seconds(), avg_duration_s,
                          remaining_datetime, humanize.naturaldelta(remaining_datetime)))

    def on_training_finished(self):
        # wait for all plotting threads to join
        print("[logger.py] Waiting for tensorboard plotting threads to join...")
        for t in self.figures_threads:
            if t is not None:
                t.join()
        # TODO write training stats
        self.tensorboard.flush()
        self.tensorboard.close()
        if self.train_config.verbosity >= 1:
            print("[RunLogger] Training has finished")

    # - - - - - - - - - - Non-threaded plots to tensorbard - - - - - - - - - -

    def plot_train_spectrograms(self, epoch, x_in, x_out, sample_info, dataset: AudioDataset,
                                model_config, train_config):
        fig, _ = utils.figures.plot_train_spectrograms(x_in, x_out, sample_info, dataset, model_config, train_config)
        self.tensorboard.add_figure('Spectrogram', fig, epoch)

    def plot_decoder_interpolation(self, epoch, generative_model, z_minibatch, sample_info, dataset: AudioDataset,
                                   output_channel_to_plot=0):
        from evaluation.interp import LatentInterpolation  # local import to prevent circular import
        interpolator = LatentInterpolation(generator=generative_model, device=z_minibatch.device)
        # z start/end tensors must be be provided as 1 x D vectors
        u, z, x = interpolator.interpolate_spectrograms_from_latent(z_minibatch[0:1, :], z_minibatch[1:2, :])
        if x.shape[1] > 1:
            x = x[:, output_channel_to_plot:output_channel_to_plot+1, :, :]
        preset_UIDs = [sample_info[i, 0].item() for i in range(2)]
        preset_names = [dataset.get_name_from_preset_UID(UID) for UID in preset_UIDs]
        title = "{} '{}' ----> {} '{}'".format(preset_UIDs[0], preset_names[0], preset_UIDs[1], preset_names[1])
        fig, _ = utils.figures.plot_spectrograms_interp(u, x, z=z, title=title, plot_delta_spectrograms=True)
        self.tensorboard.add_figure("DecoderInterpolation", fig, epoch)

    # - - - - - Multi threaded + multiprocessing plots to tensorboard - - - - -

    def plot_stats_tensorboard__threaded(self, train_config, epoch, super_metrics, ae_model):
        if self.figures_threads[0] is not None:
            if self.figures_threads[0].is_alive():
                warnings.warn("A new threaded plot request has been issued but the previous has not finished yet. "
                              "Please ignore this message if the training run is about to end.")
            self.figures_threads[0].join()
        # Data must absolutely be copied - this is multithread, not multiproc (shared data with GIL, no auto pickling)
        networks_layers_params = dict()  # If remains empty: no plot
        if epoch > 2 * train_config.beta_warmup_epochs:  # Don't plot ae_model weights histograms during first epochs
            # returns clones of layers' parameters
            networks_layers_params['Decoder'] = ae_model.decoder.get_fc_layers_parameters()
        # Launch thread using copied data
        self.figures_threads[0] = threading.Thread(target=self._plot_stats_thread,
                                                   args=(copy.deepcopy(epoch),
                                                         copy.deepcopy(super_metrics),
                                                         networks_layers_params))
        self.figures_threads[0].start()

    def _plot_stats_thread(self, epoch, super_metrics, networks_layers_params):
        if not self.use_multiprocessing:
            figs_dict = logs.logger_mp.get_stats_figures(epoch, super_metrics, networks_layers_params)
        else:
            # 'spawn' context is slower, but solves observed deadlock
            #     deadlock seems to be caused by serializing this instance
            ctx = multiprocessing.get_context('spawn')  # Utilisé au lieu de multiproc
            q = ctx.Queue()
            p = ctx.Process(target=logs.logger_mp.get_stats_figs__multiproc,
                            args=(q, epoch, super_metrics, networks_layers_params))
            p.start()
            figs_dict = q.get()  # Will block until an item is available
            p.join()
            p.close()

        for fig_name, fig in figs_dict.items():
            self.tensorboard.add_figure(fig_name, fig, epoch, close=True)

        # Those plots do not need to be here, but multi threading might help improve perfs a bit...
        if epoch > 0:
            try:
                for metric_name in ['RegOutValues/Train', 'RegOutValues/Valid']:
                    self.tensorboard.add_vector_histograms(super_metrics[metric_name], metric_name, epoch)
            except AssertionError:
                pass  # RegOut metric are not filled during pretrain, and will raise errors
        # Latent metrics: TB histograms + TB embeddings (for t-SNE and UMAP visualisations)
        self.tensorboard.add_latent_histograms(super_metrics['LatentMetric/Train'], 'Train', epoch)
        self.tensorboard.add_latent_histograms(super_metrics['LatentMetric/Valid'], 'Valid', epoch)
        # Validation embeddings only (Train embeddings tensor is very large, slow download and analysis)
        self.tensorboard.add_latent_embedding(super_metrics['LatentMetric/Valid'], 'Valid', epoch)
        # Network weights histograms
        for network_name, network_layers in networks_layers_params.items():  # key: e.g. 'Decoder'
            for layer_name, layer_params in network_layers.items():  # key: e.g. 'FC0'
                for param_name, param_values in layer_params.items():  # key: e.g. 'weight_abs'
                    # Default bins estimator: 'tensorflow'. Other estimator: 'fd' (robust for outliers)
                    self.tensorboard.add_histogram('{}/{}/{}'.format(network_name, layer_name, param_name),
                                                   param_values, epoch)
                    self.tensorboard.add_histogram('{}_no_outlier/{}/{}'.format(network_name, layer_name, param_name),
                                                   utils.stat.remove_outliers(param_values), epoch)





