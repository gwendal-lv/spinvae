"""
Contains base classes (abstract or not) for models.
"""

import torch
import torch.nn as nn



class TrainableModel(nn.Module):
    def __init__(self, train_config=None):
        super().__init__()
     # TODO init

    @property
    def optimizer(self):
        return None

    @property
    def scheduler(self):
        return None

    # TODO méthodes de load/save from checkpoint



class DummyModel(nn.Module):
    def __init__(self, output_size=(1, )):
        """
        An empty neural network, which can be used to do nothing (but supports usual neural network calls).
        (e.g. when pre-training a sub-network from a bigger structure, but the whole structure is needed)
        """
        super().__init__()
        self.output_size = output_size

    def forward(self, x: torch.Tensor):
        return torch.zeros(self.output_size, device=x.device)


