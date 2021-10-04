"""
Contains base classes (abstract or not) for models.
"""

import torch
import torch.nn as nn



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

