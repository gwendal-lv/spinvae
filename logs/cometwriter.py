from typing import Optional

import numpy as np
import torch
from comet_ml import Experiment  # Needs to be imported first

import matplotlib.pyplot as plt

import utils.stat
from logs.metrics import LatentMetric


class CometWriter:
    def __init__(self, model_config, train_config):  # TODO typing hint after config.py refactoring
        # We'll create a new experiment, or load an existing one if restarting from a checkpoint
        if train_config.start_epoch == 0:
            self.experiment = Experiment(
                api_key=model_config.comet_api_key, workspace=model_config.comet_workspace,
                project_name=model_config.comet_project_name,
                log_git_patch=False,  # Quite large, and useless during remote dev (different commit on remote machine)
            )
            self.experiment.set_name(model_config.name + '/' + model_config.run_name)
            # TODO refactor - MODEL CONFIG IS MODIFIED HERE - dirty but quick....
            model_config.comet_experiment_key = self.experiment.get_key()
        else:
            raise NotImplementedError()  # TODO implement ExistingExperiment

    def log_metric_with_context(self, name: str, value):
        """ Logs a scalar metric, and tries to set a comet context before doing it (e.g. "Loss/Train" will be
            logged as "Loss" in a train experiment context). """
        if self._has_train_suffix(name):
            with self.experiment.train():
                self.experiment.log_metric(self._remove_train_valid_suffix(name), value)
        elif self._has_validation_suffix(name):
            with self.experiment.validate():
                self.experiment.log_metric(self._remove_train_valid_suffix(name), value)
        else:  # Log without context (e.g. scheduler, ...)
            self.experiment.log_metric(name, value)

    def log_figure_with_context(self, name: str, fig, step: Optional[int] = None):
        """ Logs a MPL figure, and tries to set a comet context before doing it (see log_metric_with_context).
            Step can be forced if this method is called asynchronously (threaded plots). """
        if self._has_train_suffix(name):
            with self.experiment.train():
                self.experiment.log_figure(self._remove_train_valid_suffix(name), fig, step=step)
        elif self._has_validation_suffix(name):
            with self.experiment.validate():
                self.experiment.log_figure(self._remove_train_valid_suffix(name), fig, step=step)
        else:  # Log without context (e.g. scheduled hyper-parameters (LR, gamma, ...), ...)
            self.experiment.log_figure(name, fig, step=step)
        plt.close(fig)

    @staticmethod
    def _assert_is_train_or_valid_dataset_type(ds_type: str):
        """ Returns True if this is a training dataset, False if it is a validation dataset, raises an exception
            otherwise. """
        if "train" in ds_type.lower():
            return True
        elif "valid" in ds_type.lower():
            return False
        else:
            raise ValueError("Dataset type '{}' cannot be identified as a training or a validation dataset.")


    @staticmethod
    def _has_train_suffix(name: str):
        return name.endswith("/Train") or name.endswith("/train")

    @staticmethod
    def _has_validation_suffix(name: str):
        return name.endswith("/Valid") or name.endswith("/valid") \
               or name.endswith("/Validation") or name.endswith("/validation")

    @staticmethod
    def _remove_train_valid_suffix(s: str, suffix_separator='/'):
        s_split = s.split(suffix_separator)
        s_split.pop()
        return suffix_separator.join(s_split)

    def log_latent_histograms(self, latent_metric: LatentMetric, dataset_type: str,
                              epoch: Optional[int] = None, step: Optional[int] = None):
        """
        Adds histograms related to z0 and zK samples to Tensorboard.

        :param dataset_type: 'Train', 'Valid', ...
        :param epoch: Optional epoch, should be used with async (threaded) plot rendering.
        :param step: Optional step, should be used with async (threaded) plot rendering.
        """
        dataset_type = dataset_type.lower()
        if dataset_type != 'train' and dataset_type != 'valid':
            raise ValueError("Only 'train' and 'valid' dataset types are supported by this method.")
        with self.experiment.train() if dataset_type == 'train' else self.experiment.validate():
            z0 = latent_metric.get_z('z0').flatten()
            zK = latent_metric.get_z('zK').flatten()
            self.experiment.log_histogram_3d(z0, name='z0', step=step, epoch=epoch)
            self.experiment.log_histogram_3d(zK, name='zK', step=step, epoch=epoch)

    def log_latent_embedding(self, latent_metric: LatentMetric, dataset_type: str, epoch: int):
        rng = np.random.default_rng(seed=epoch)  # TODO refactor this, move to utils/label.py
        labels_uint8 = latent_metric.get_z('label')  # One sample can have 0, 1 or multiple labels
        # labels converted to strings
        labels = list()
        for i in range(labels_uint8.shape[0]):  # np.nonzero does not have a proper row-by-row option
            label_indices = np.flatnonzero(labels_uint8[i, :])
            if len(label_indices) == 0:
                # labels.append("No label")
                labels.append(-1)
            else:  # Only 1 label will be displayed (randomly chosen if multiple labels)
                # TODO try add multiple labels ? or use metadata to build a secondary label ?
                rng.shuffle(label_indices)  # in-place
                # labels.append(str(label_indices[0]))  # FIXME use string label instead of int index
                labels.append(label_indices[0])
        # labels = nup(labels)  # TODO don't send torch tensors ?
        # TODO add z0 and zK embeddings
        with self.experiment.train() if self._assert_is_train_or_valid_dataset_type(dataset_type) \
                else self.experiment.validate():
            self.experiment.log_embedding(latent_metric.get_z('z0'), labels, title="z0")
            print("LOGGED Z0")  # FIXME stuck here ??? TOO BIG ?

