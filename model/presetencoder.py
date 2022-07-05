
import warnings

import torch
import torch.nn as nn
import torchinfo

from data.preset2d import Preset2dHelper
from model.presetmodel import parse_preset_model_architecture, PresetEmbedding


class PresetEncoder(nn.Module):
    # TODO EXPECTED OUTPUT SHAPE CTOR ARG
    def __init__(self, architecture: str, hidden_size: int, preset_helper: Preset2dHelper):
        super().__init__()
        self.arch = parse_preset_model_architecture(architecture)
        self.preset_helper = preset_helper
        self.embedding = PresetEmbedding(hidden_size, preset_helper)

        # TODO MLP encoder

        warnings.warn("TODO: Preset encoder not implemented yet")

    def forward(self, u_in: torch.Tensor):
        embed = self.embedding(u_in)
        return embed  # FIXME TODO

    def get_summary(self, minibatch_size=1):
        u = self.preset_helper.get_null_learnable_preset(minibatch_size)
        return torchinfo.summary(
            self, input_data=u,
            depth=5, verbose=0, device=torch.device('cpu'),
            col_names=("input_size", "kernel_size", "output_size", "num_params", "mult_adds"),
            row_settings=("depth", "var_names")
        )
