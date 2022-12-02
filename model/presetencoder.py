
import warnings
from typing import List

import numpy as np
import torch
import torch.nn as nn
import torchinfo

from data.preset2d import Preset2dHelper
from model.presetmodel import parse_preset_model_architecture, PresetEmbedding, get_act, get_transformer_act


class PresetEncoder(nn.Module):
    def __init__(self, architecture: str, hidden_size: int, preset_helper: Preset2dHelper,
                 dim_z: int, output_fm_shape: List[int], dropout_p=0.0):
        super().__init__()
        self.arch = parse_preset_model_architecture(architecture)
        self.arch_args = self.arch['args']
        self.n_layers = self.arch['n_layers']
        self.preset_helper = preset_helper
        self.hidden_size = hidden_size
        self.output_fm_shape = output_fm_shape

        self.tfm, self.lstm = None, None
        if self.arch['name'] in ['tfm', 'lstm']:
            if self.arch['name'] == 'tfm':
                # The Transformer encoder is much simpler than the decoder: pure parallel, feedforward, single pass
                n_head = 4
                tfm_layer = nn.TransformerEncoderLayer(
                    self.hidden_size, n_head, dim_feedforward=self.hidden_size*4,
                    batch_first=True, dropout=dropout_p, activation=get_transformer_act(self.arch_args)
                )
                self.tfm = nn.TransformerEncoder(tfm_layer, self.n_layers)  # No output norm
                self.n_output_tokens = 2 * dim_z // hidden_size

            elif self.arch['name'] == 'lstm':
                self.lstm = nn.LSTM(
                    input_size=self.hidden_size, hidden_size=self.hidden_size, num_layers=self.n_layers,
                    bidirectional=True, batch_first=True, dropout=dropout_p
                )
                # Outputs could have the right size; however, we should not force the forward or backward pass
                # to represent mu or sigma, respectively. So this MLP allows to "shuffle" the final hidden values.
                self.lstm_out_mlp = nn.Sequential(
                    nn.Linear(4 * self.hidden_size, 4 * self.hidden_size),
                    get_act(self.arch_args),
                    nn.Linear(4 * self.hidden_size, 2 * self.hidden_size),
                )

            # Declared after the transformer - allows perfect reproducibility of results from oct. 2022
            max_norm = np.sqrt(self.hidden_size) if self.arch_args['embednorm'] else None
            self.embedding = PresetEmbedding(hidden_size, preset_helper, max_norm)

        elif self.arch['name'] == 'mlp':
            self.mlp = nn.Sequential()
            n_hidden_units = 2048
            reduced_hidden_size = n_hidden_units // preset_helper.n_learnable_params
            self.embedding = PresetEmbedding(reduced_hidden_size, preset_helper)
            n_mlp_input_units = reduced_hidden_size * preset_helper.n_learnable_params
            n_mlp_output_units = 2 * dim_z  # tfm and mlp basically output a mu&var - like vector
            for l in range(0, self.n_layers):
                if l > 0:
                    if l < self.n_layers - 1:  # No norm before the last fc
                        if self.arch_args['bn']:
                            self.mlp.add_module('bn{}'.format(l), nn.BatchNorm1d(n_hidden_units))
                        if self.arch_args['ln']:
                            raise NotImplementedError()
                    self.mlp.add_module('act{}'.format(l), get_act(self.arch_args))
                    if dropout_p > 0.0 and (l < self.n_layers - 1):  # No dropout before the last fc
                        self.mlp.add_module('drop{}'.format(l), nn.Dropout(dropout_p))
                n_in_features = n_hidden_units if (l > 0) else n_mlp_input_units
                n_out_features = n_hidden_units if (l < self.n_layers - 1) else n_mlp_output_units
                self.mlp.add_module('fc{}'.format(l), nn.Linear(n_in_features, n_out_features))

        else:
            raise ValueError("Unavailable architecture '{}'".format(self.arch['name']))

        # Reshaping is always the same
        assert (2 * dim_z) % (output_fm_shape[1] * output_fm_shape[2]) == 0
        n_reshape_ch = (2 * dim_z) // (output_fm_shape[1] * output_fm_shape[2])
        self.u_pre_out_shape = (n_reshape_ch, output_fm_shape[1], output_fm_shape[2])
        # But the number of channels (in the obtained 2D feature maps) might be increased if required
        if np.prod(output_fm_shape) == (2 * dim_z):  # Output mu and sigma directly: a simple reshape will be required
            self.extend_ch_conv = None
            assert self.u_pre_out_shape == tuple(output_fm_shape)
        elif np.prod(output_fm_shape) > (2 * dim_z):  # Output must be reshaped then num. ch. increased
            self.extend_ch_conv = nn.Conv2d(n_reshape_ch, output_fm_shape[0], 1)
        else:
            raise AssertionError()

    def forward(self, u_in: torch.Tensor):
        N = u_in.shape[0]

        if self.tfm is not None:  # Transformer
            embed = self.embedding(u_in, n_special_end_tokens=self.n_output_tokens)
            u_hidden = self.tfm(embed)
            # retrieve only the last tokens (compressed latent preset representation)
            #   discard hidden representations of each individual synth parameter
            u_hidden = u_hidden[:, -self.n_output_tokens:, :]

        elif self.lstm is not None:  # Bi-LSTM
            embed = self.embedding(u_in, n_special_end_tokens=0)  # seq2seq-like model: compute latent from final h, c
            u_output, (h, c) = self.lstm(embed)
            h, c = torch.transpose(h, 0, 1), torch.transpose(c, 0, 1)  # set batch first (not automatic)
            # Doc is not unambiguous.... https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html
            # We'll guess that h, c concatenation is done forward/backward first, then concatenate all layers' hidden.
            # So last layer's h, c should be:
            last_h, last_c = h[:, -2:, :], c[:, -2:, :]
            u_hidden = torch.stack([last_h, last_c], dim=1).view(N, -1)
            u_hidden = self.lstm_out_mlp(u_hidden)

        else:  # MLP
            embed = self.embedding(u_in, pos_embed=False)  # small embeds, shouldn't use pos embed anyway
            u_hidden = self.mlp(embed.view(N, -1))  # Remove any position information, apply MLP

        u_hidden = u_hidden.view(N, *self.u_pre_out_shape)
        if self.extend_ch_conv is not None:
            u_hidden = self.extend_ch_conv(u_hidden)
        return u_hidden

    def get_summary(self, minibatch_size=1):
        u = self.preset_helper.get_null_learnable_preset(minibatch_size)
        return torchinfo.summary(
            self, input_data=u,
            depth=6, verbose=0, device=torch.device('cpu'),
            col_names=("input_size", "output_size", "num_params", "mult_adds"),
            row_settings=("depth", "var_names")
        )
