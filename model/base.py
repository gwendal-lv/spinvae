"""
Contains base classes (abstract or not) for models.
"""
import pathlib
import warnings
from abc import abstractmethod
from typing import Optional, Tuple, List, Dict
import copy

import numpy as np
import torch
import torch.nn as nn

import config


def build_optimizer(train_config: config.TrainConfig, lr: float, parameters):
    if train_config.optimizer.lower() == 'adam':
        return torch.optim.Adam(
            parameters, lr=lr, weight_decay=train_config.weight_decay, betas=train_config.adam_betas
        )
    else:
        raise NotImplementedError("Optimizer '{}' not available.".format(train_config.optimizer))


def build_scheduler(train_config: config.TrainConfig, optimizer):
    if train_config.scheduler_name.lower() == 'reducelronplateau':
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            factor=train_config.scheduler_lr_factor,
            patience=train_config.scheduler_patience,
            cooldown=train_config.scheduler_cooldown,
            threshold=train_config.scheduler_threshold,
            verbose=(train_config.verbosity >= 2)
        )
    elif train_config.scheduler_name.lower() == 'steplr':
        return torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=train_config.scheduler_period,
            gamma=train_config.scheduler_lr_factor
        )
    else:
        raise NotImplementedError("Scheduler '{}' not available.".format(train_config.scheduler_name))


def get_optimizer_lr(optimizer):
    """ Returns the optimizer's learning rate (which should be the same for all param groups). """
    lrs = list()
    for param_group in optimizer.param_groups:
        lrs.append(param_group['lr'])
    lrs = set(lrs)
    if len(lrs) > 1:
        warnings.warn("{} different learning rates were found. The average learning rate will be returned. LRs: {}"
                      .format(len(lrs), lrs))
        return np.asarray(list(lrs)).mean()
    else:
        return list(lrs)[0]


def set_optimizer_lr(optimizer, lr):
    # https://discuss.pytorch.org/t/change-learning-rate-in-pytorch/14653
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


class TrainableModel(nn.Module):
    def __init__(self, train_config: Optional[config.TrainConfig] = None, model_type: Optional[str] = None):
        """
        A base class for any model that integrates an optimizer (e.g. Adam) and a scheduler (e.g. LR reduction on
        loss plateau).

        :param train_config: train attribute from the config.py file
        :param model_type: 'ae' or 'reg' to build a truly trainable model; or None to build to a dummy trainable model.
        """
        super().__init__()
        self.train_config, self.model_type = copy.deepcopy(train_config), model_type
        if self.train_config is not None:
            if self.model_type is None:
                raise ValueError("If train_config arg is given, then model_type must be provided as well.")
        else:
            self.model_type = None  # Forced model type to None (dummy trainable model)
        self._optimizer = None
        self._scheduler = None

    def init_optimizer_and_scheduler(self):
        """ Init must be done after construction (all sub-modules' parameters must be available).
        Does nothing if no train_config has been given to this instance. """
        self.train()
        # Optimizer
        if self.model_type is not None:
            self._optimizer = build_optimizer(
                self.train_config, self.train_config.initial_learning_rate[self.model_type], self.parameters())
        else:
            self._optimizer = torch.optim.SGD(self.parameters(), lr=1e-4)
        # scheduler
        if self.model_type is not None:
            self._scheduler = build_scheduler(self.train_config, self._optimizer)
        else:
            self._scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer,
                                                                         patience=10**12, cooldown=10**12)
        self.eval()

    @property
    def optimizer(self):
        return self._optimizer

    @property
    def learning_rate(self):
        return get_optimizer_lr(self.optimizer)

    @learning_rate.setter
    def learning_rate(self, lr):
        """ The learning rate to be assigned to each param group of this model's optimizer. """
        set_optimizer_lr(self.optimizer, lr)

    @property
    def scheduler(self):
        return self._scheduler

    def load_checkpoint(self, checkpoint, eval_only=False):
        if self.model_type is None:
            raise AssertionError("Cannot load a checkpoint for this dummy None model.")
        # The checkpoint might contain info for different model types
        sub_checkpoint = checkpoint[self.model_type]
        # Dict keys are defined in logs/logger.py, RunLogger::save_checkpoint(...)
        self.load_state_dict(sub_checkpoint['model_state_dict'])
        if not eval_only:
            self.optimizer.load_state_dict(sub_checkpoint['optimizer_state_dict'])
            # FIXME scheduler might be different after loading the model
            self.scheduler.load_state_dict(sub_checkpoint['scheduler_state_dict'])

    def get_detailed_summary(self) -> str:
        raise NotImplementedError()


class DummyModel(TrainableModel):
    def __init__(self, output_size=(1, )):
        """
        An empty neural network, which can be used to do nothing (but supports usual neural network calls).
        (e.g. when pre-training a sub-network from a bigger structure, but the whole structure is needed)
        """
        super().__init__(train_config=None, model_type=None)
        self.output_size = output_size
        self._dummy_layer = nn.Linear(2, 2)

    def forward(self, x: torch.Tensor):
        return torch.zeros(self.output_size, device=x.device)


class DummyRegModel(DummyModel):
    def precompute_u_in_permutations(self, u_in):
        pass

    def precompute_u_out_with_symmetries(self, u_out):
        pass


# ---------- New base classes for Hierarchical models with combined audio and presets ----------

class TrainableMultiGroupModel(nn.Module):
    def __init__(self, train_config: config.TrainConfig,
                 param_group_names: List[str], trained_param_group_names: List[str]):
        """
        Base class for a model which contain groups of modules (parameters) which require different optimizers
        and scheduler.
        We don't use the named_groups feature from PyTorch optimizers, because some groups (sub-nets)
        can be different before and after pre-train.

        This class also handles saving and loading checkpoints for each group and associated optimizer and
        scheduler.
        """
        super().__init__()
        self._train_config = copy.deepcopy(train_config)
        # Opt and Sched won't be created for non-trained groups
        self.param_group_names, self.trained_param_group_names = param_group_names, trained_param_group_names
        self._optimizers = {k: None for k in param_group_names}
        self._schedulers = {k: None for k in param_group_names}

    @property
    def pre_training_audio(self):
        return self._train_config.pretrain_audio_only

    @property
    def params_loss_compensation_factor(self):
        return self._train_config.params_loss_compensation_factor

    @abstractmethod
    def get_custom_group_module(self, group_name: str) -> nn.Module:
        pass

    def _init_optimizers_and_schedulers(self):
        for k in self.trained_param_group_names:
            self._optimizers[k] = build_optimizer(
                self._train_config, self._train_config.initial_learning_rate[k],
                self.get_custom_group_module(k).parameters()
            )
        for k in self.trained_param_group_names:
            self._schedulers[k] = build_scheduler(self._train_config, self._optimizers[k])

    def set_warmup_lr_factor(self, lr_factor: float):
        """ Enforces LR values for all available optimizers, given lr_factor typically in [0.0, 1.0]. """
        for k in self.trained_param_group_names:
            set_optimizer_lr(self._optimizers[k], self._train_config.initial_learning_rate[k] * lr_factor)

    def get_group_lr(self, group_name: str):
        return get_optimizer_lr(self._optimizers[group_name])

    def optimizers_zero_grad(self):
        """ Applies only to the actually used opts and scheds (not all of them during pretrain) """
        for k in self.trained_param_group_names:
            self._optimizers[k].zero_grad()

    def optimizers_step(self):
        for k in self.trained_param_group_names:
            self._optimizers[k].step()

    def schedulers_step(self, loss_values: Optional[Dict[str, float]] = None):
        for k in self.trained_param_group_names:
            if isinstance(self._schedulers[k], torch.optim.lr_scheduler.ReduceLROnPlateau):
                self._schedulers[k].step(loss_values[k])
            else:
                self._schedulers[k].step()

    def save_checkpoints(self, model_dir: pathlib.Path):
        """ Saves all sub-models into a unique checkpoint.tar file (non-trained sub-models will be saved as None). """
        checkpoint_dict = dict()
        for k in self.param_group_names:
            checkpoint_dict[k] = dict()
            if k in self.trained_param_group_names:  # Save trained groups only
                checkpoint_dict[k] = {
                    'model_state_dict': self.get_custom_group_module(k).state_dict(),
                    'optimizer_state_dict': self._optimizers[k].state_dict(),
                    'scheduler_state_dict': self._schedulers[k].state_dict()
                }
            else:  # Save non-trained parameters as None
                checkpoint_dict[k] = {
                    'model_state_dict': None, 'optimizer_state_dict': None, 'scheduler_state_dict': None
                }
        torch.save(checkpoint_dict, model_dir.joinpath("checkpoint.tar"))
        if self._train_config.verbosity >= 1:
            print("[RunLogger] Saved checkpoint (models, optimizers, schedulers) to {}"
                  .format(model_dir.joinpath("checkpoint.tar")))

    def load_checkpoints(self, checkpoints_path: pathlib.Path, reload_opt_sched=False, map_location=None):
        """ Loads weights from pre-trained submodels (dicts corresponding to non-pretrained sub-models are set
        to None values). """
        checkpoint_dict = torch.load(checkpoints_path, map_location=map_location)
        for k, submodel_checkpoint in checkpoint_dict.items():
            assert k in self.param_group_names  # e.g. 'audio', 'latent' or 'preset'
            if submodel_checkpoint is not None and submodel_checkpoint['model_state_dict'] is not None:
                group_module = self.get_custom_group_module(k)
                group_module.load_state_dict(submodel_checkpoint['model_state_dict'])
                if self._train_config.verbosity >= 1:
                    print("[TrainableMultiGroupModel] Loaded '{}' model_state_dict from {}".format(k, checkpoints_path))
                if reload_opt_sched:  # FIXME TODO???
                    raise NotImplementedError("Can only load some models at the moment (not the optimizers or scheds")

