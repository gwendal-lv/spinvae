"""
Contains base classes (abstract or not) for models.
"""
import warnings
from typing import Optional
import copy

import numpy as np
import torch
import torch.nn as nn



class TrainableModel(nn.Module):
    def __init__(self, train_config=None, model_type: Optional[str] = None):
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
            if self.train_config.optimizer.lower() == 'adam':
                self._optimizer = torch.optim.Adam(self.parameters(),
                                                   lr=self.train_config.initial_learning_rate[self.model_type],
                                                   weight_decay=self.train_config.weight_decay,
                                                   betas=self.train_config.adam_betas)
            else:
                raise NotImplementedError("Optimizer '{}' not available.".format(self.train_config.optimizer))
            # Construction finished
        else:
            self._optimizer = torch.optim.SGD(self.parameters(), lr=1e-4)
        # scheduler
        if self.model_type is not None:
            if self.train_config.scheduler_name.lower() == 'reducelronplateau':
                self._scheduler = torch.optim.lr_scheduler.\
                        ReduceLROnPlateau(self.optimizer, factor=self.train_config.scheduler_lr_factor[self.model_type],
                                          patience=self.train_config.scheduler_patience[self.model_type],
                                          cooldown=self.train_config.scheduler_cooldown[self.model_type],
                                          threshold=self.train_config.scheduler_threshold,
                                          verbose=(self.train_config.verbosity >= 2))
            else:
                raise NotImplementedError("Scheduler '{}' not available.".format(self.train_config.scheduler_name))
        else:
            self._scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer,
                                                                         patience=10**12, cooldown=10**12)
        self.eval()

    @property
    def optimizer(self):
        return self._optimizer

    @property
    def learning_rate(self):
        """ The current optimizer's learning rate (which should be the same for all param groups). """
        lrs = list()
        for param_group in self.optimizer.param_groups:
            lrs.append(param_group['lr'])
        lrs = set(lrs)
        if len(lrs) > 1:
            warnings.warn("{} different learning rates were found. The average learning rate will be returned. LRs: {}"
                          .format(len(lrs), lrs))
            return np.asarray(list(lrs)).mean()
        else:
            return list(lrs)[0]

    @learning_rate.setter
    def learning_rate(self, lr):
        """ The learning rate to be assigned to each param group of this model's optimizer. """
        # https://discuss.pytorch.org/t/change-learning-rate-in-pytorch/14653
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    @property
    def scheduler(self):
        return self._scheduler

    # TODO méthodes de load/save from checkpoint



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


