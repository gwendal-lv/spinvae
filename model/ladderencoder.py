from typing import Dict, Tuple, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchinfo
import warnings

from model.ladderbase import LadderBase
from model.convlayer import ConvBlock2D, DownsamplingResBlock, SelfAttentionConv2D
from model.convlstm import ConvLSTM


class LadderEncoder(LadderBase):

    def __init__(self, conv_arch, latent_arch, latent_levels: int, input_tensor_size: Tuple[int, int, int, int],
                 approx_dim_z=2000):
        """
        Contains cell which define the hierarchy levels (the output of each cell is used to extract latent values)
            Each cell is made of blocks (skip connection may be added/concat/other at the end of block)
                Each block contains one or more conv, act and norm layers
         TODO also encode optional preset

        :param latent_levels:
        :param input_tensor_size:
        :param approx_dim_z:
        """

        super().__init__(conv_arch, latent_arch)
        self._input_tensor_size = input_tensor_size
        self._single_ch_input_size = (1, 1, input_tensor_size[2], input_tensor_size[3])
        self._audio_seq_len = input_tensor_size[1]

        # - - - - - 1) Build the single-channel CNN (applied to each input audio channel) - - - - -
        conv_args = self.single_ch_conv_arch['args']
        n_blocks = self.single_ch_conv_arch['n_blocks']
        if self.single_ch_conv_arch['name'].startswith('specladder'):
            assert n_blocks == 8  # 9 layers would allow to set dim_z exactly but is not implemented yet
            if conv_args['adain']:
                raise NotImplementedError()
            if conv_args['att'] and self.single_ch_conv_arch['n_layers_per_block'] < 2:
                raise ValueError("'_att' conv arg will add a residual self-attention layer and requires >= 2 layers")
            self.single_ch_cells = list()
            if latent_levels == 1:
                cells_last_block = [n_blocks - 1]  # Unique cell ends with the very last block
            elif latent_levels == 2:
                cells_last_block = [4, n_blocks - 1]
            elif latent_levels == 3:
                cells_last_block = [3, 5, n_blocks - 1]
            elif latent_levels == 4:
                cells_last_block = [2, 3, 5, n_blocks - 1]
            else:
                raise NotImplementedError("Cannot build encoder with {} latent levels".format(latent_levels))
            self.single_ch_cells.append(nn.Sequential())

            for i_blk in range(n_blocks):
                residuals_path = nn.Sequential()
                blk_in_ch = 2**(i_blk+2)  # number of input channels
                blk_out_ch = 2**(i_blk+3)  # number of channels increases after each strided conv (at the block's end)
                if conv_args['big']:
                    blk_in_ch, blk_out_ch = blk_in_ch * 2, blk_out_ch * 2
                min_ch = 1 if not conv_args['bigger'] else 128
                max_ch = 512 if not conv_args['bigger'] else 1024
                blk_in_ch, blk_out_ch = np.clip([blk_in_ch, blk_out_ch], min_ch, max_ch)
                blk_hid_ch = blk_in_ch  # base number of internal (hidden) block channels

                if i_blk == 0:  # First block: single conv channel
                    strided_conv_block = ConvBlock2D(nn.Conv2d(1, blk_out_ch, (5, 5), 2, 2), None, None, 'c')
                    residuals_path.add_module('strided', strided_conv_block)
                else:  # Other block can contain multiple conv block
                    for j in range(self.single_ch_conv_arch['n_layers_per_block']-1):
                        # first layer can be self-attention w/ 2D positional encodings
                        if j == 0 and conv_args['att'] and (3 <= i_blk <= 5):
                            self_att = SelfAttentionConv2D(blk_hid_ch, position_encoding=True)
                            residuals_path.add_module('self_att', ConvBlock2D(
                                self_att, None, self._get_conv_norm(blk_hid_ch), 'nc'))
                        else:
                            # No depth-separable conv in the encoder (see NVAE NeurIPS 2020) - only 3x3 conv
                            conv = nn.Conv2d(blk_hid_ch, blk_hid_ch, (3, 3), 1, 1)
                            residuals_path.add_module('conv' + str(j), ConvBlock2D(
                                conv, self._get_conv_act(), self._get_conv_norm(blk_hid_ch), 'nac'))
                    strided_conv = nn.Conv2d(blk_hid_ch, blk_out_ch, (4, 4), 2, 2)
                    residuals_path.add_module('strided', ConvBlock2D(
                        strided_conv, self._get_conv_act(), self._get_conv_norm(blk_hid_ch), 'nac'))

                # Add a skip-connection if required, then add this new block to the current cell
                if conv_args['res'] and i_blk > 0:
                    current_block = DownsamplingResBlock(residuals_path)
                else:
                    current_block = residuals_path  # No skip-connection
                self.single_ch_cells[-1].add_module('blk{}'.format(i_blk), current_block)
                if i_blk in cells_last_block and i_blk < (n_blocks - 1):  # Start building the next cell
                    self.single_ch_cells.append(nn.Sequential())

        else:  # Spectrograms only are supported at the moment
            raise NotImplementedError("Unimplemented '{}' architecture".format(self.single_ch_conv_arch['name']))

        # - - - - - 2) Latent inference networks (1 / cell) - - - - -
        self.latent_cells = list()
        # retrieve output size of each CNN cell and approximate dim_z for each dimension
        self.cells_output_shapes = self._get_cells_output_shapes()
        # Latent space size: number of channels chosen such that the total num of latent coordinates
        # is close the approx_dim_z_per_level value (these convolutions keep feature maps' H and W)
        n_latent_ch_per_level = self._get_conv_latent_ch_per_level(approx_dim_z)

        # Conv alone: for a given level, feature maps from all input spectrograms are merged using a conv network only
        # and the output is latent distributions parameters mu and sigma^2. Should be a very simple and fast network
        if self.latent_arch['name'] == "conv" or self.latent_arch['name'] == 'lstm':
            use_lstm = self.latent_arch['name'] == 'lstm'
            if self.latent_arch['args']['k1x1']:
                kernel_size, padding = (1, 1), (0, 0)
            elif self.latent_arch['args']['k3x3']:
                kernel_size, padding = (3, 3), (1, 1)
            else:
                raise NotImplementedError("Can't build latent cells: conv kernel arg ('_k1x1' or '_k3x3') not provided")
            for i, cell_output_shape in enumerate(self.cells_output_shapes):
                n_latent_ch = n_latent_ch_per_level[i] * 2  # Output mu and sigma2
                cell_args = (
                    cell_output_shape[1], cell_output_shape[2:], self._audio_seq_len,
                    n_latent_ch, self.latent_arch['n_layers'], kernel_size, padding, self.latent_arch['args']
                )
                if use_lstm:
                    self.latent_cells.append(ConvLSTMLatentCell(*cell_args))
                else:
                    self.latent_cells.append(ConvLatentCell(*cell_args))
        else:
            raise NotImplementedError("Cannot build latent arch {}: name not implemented".format(latent_arch))

        # Finally, Python lists must be converted to nn.ModuleList to be properly recognized by PyTorch
        self.single_ch_cells = nn.ModuleList(self.single_ch_cells)
        self.latent_cells = nn.ModuleList(self.latent_cells)

    def _get_conv_latent_ch_per_level(self, approx_dim_z: int):
        """ Computes the number of output latent channels for each level, such that latent feature maps keep
         a spatial structure (2D multichannel images, different resolutions) and such that the total number
         of pixels is very close to approx_dim_z. """
        latent_ch_per_level = list()
        # Build a linearly decreasing number of latent coords for the most hidden layers, even if the
        # feature maps' sizes decreases quadratically - not to favor too much the shallowest feature maps, which
        # could (seems to) lead to overfitting
        dim_z_ratios = np.arange(len(self.single_ch_cells), 0, -1)
        dim_z_ratios = dim_z_ratios / dim_z_ratios.sum()
        remaining_z_dims = approx_dim_z  # Number of latent dims yet to be assigned to a latent level
        # Assign the exact number of channels, level-by-level, to get as close as possible to the requested dim_z
        for latent_lvl, conv_cell_output_shape in enumerate(self.cells_output_shapes):  # Enc: latent_lvl == cell_index
            num_values_per_ch = np.prod(conv_cell_output_shape[2:])
            if latent_lvl < (len(self.cells_output_shapes) - 1):  # For all latent levels but the deepest one
                num_ch = round(approx_dim_z * dim_z_ratios[latent_lvl] / num_values_per_ch)
                # Update the ratios using the actually obtained ratio for this level
                current_z_dims = num_ch * num_values_per_ch
                dim_z_ratios[latent_lvl] = current_z_dims / approx_dim_z
                sum_set_ratios = np.sum(dim_z_ratios[0:latent_lvl+1])
                sum_free_ratios = np.sum(dim_z_ratios[latent_lvl+1:])
                dim_z_ratios[latent_lvl+1:] = dim_z_ratios[latent_lvl+1:] * (1 - sum_set_ratios) / sum_free_ratios
                remaining_z_dims -= current_z_dims
            else:  # use remaining_z_dims for the deepest latent level (smallest feature maps, easier adjustements)
                num_ch = round(remaining_z_dims / num_values_per_ch)
            if num_ch < 1:
                raise ValueError(
                    "Approximate requested dim_z is too low (cell {} output shape{}). Please increase the requested"
                    " latent dimension (current: {}).".format(latent_lvl, conv_cell_output_shape, approx_dim_z))
            latent_ch_per_level.append(num_ch)
        return latent_ch_per_level

    def get_custom_param_group(self, group_name: str):
        if group_name == 'audio':
            return self.single_ch_cells.parameters()
        elif group_name == 'latent':
            return self.latent_cells.parameters()
        elif group_name == 'preset':
            return list()
        else:
            raise ValueError("Unavailable group_name '{}'".format(group_name))

    def forward(self, x, u=None, midi_notes=None):
        """ Returns (z_mu, z_var): lists of length latent_levels """
        # 1) Apply single-channel CNN to all input channels
        latent_cells_audio_input_tensors = [[] for _ in self.latent_cells]  # 1st dim: latent level ; 2nd dim: input ch
        for ch in range(self._input_tensor_size[1]):  # Apply all cells to a channel
            cell_x = torch.unsqueeze(x[:, ch, :, :], dim=1)
            for latent_level, cell in enumerate(self.single_ch_cells):
                cell_x = cell(cell_x)
                latent_cells_audio_input_tensors[latent_level].append(cell_x)
        # Latent levels are currently independent (no top-down conditional posterior or prior)
        # We just stack inputs from all input channels to create the sequence dimension
        latent_cells_audio_input_tensors = [torch.stack(t, dim=1) for t in latent_cells_audio_input_tensors]
        # 2) Compute latent vectors: tuple (mean and variance) of lists (one tensor per latent level)
        z_mu, z_var = list(), list()
        for latent_level, latent_cell in enumerate(self.latent_cells):
            z_out = latent_cell(latent_cells_audio_input_tensors[latent_level], u, midi_notes)
            n_ch = z_out.shape[1]
            z_mu.append(z_out[:, 0:n_ch//2, :, :])
            z_var.append(F.softplus(z_out[:, n_ch//2:, :, :]))
        return z_mu, z_var

    @property
    def _summary_col_names(self):
        return "input_size", "kernel_size", "output_size", "num_params", "mult_adds"

    def get_single_ch_conv_summary(self):
        """ Torchinfo summary of the CNN that is sequentially applied to each input spectrogram. Does not
         include the output to extract latent vectors. """
        single_ch_cnn_without_latent = nn.Sequential()
        for i, cell in enumerate(self.single_ch_cells):
            single_ch_cnn_without_latent.add_module('cell{}'.format(i), cell)
        with torch.no_grad():
            return torchinfo.summary(
                single_ch_cnn_without_latent, input_size=self._single_ch_input_size,
                depth=5, verbose=0, device=torch.device('cpu'),
                col_names=self._summary_col_names, row_settings=("depth", "var_names")
            )

    def get_latent_cells_summaries(self):
        summaries = dict()
        for i, latent_cell in enumerate(self.latent_cells):
            latent_cell_input_shape = list(self.cells_output_shapes[i])
            latent_cell_input_shape.insert(1, self._audio_seq_len)
            input_data = {'x_audio': torch.zeros(latent_cell_input_shape),  # FIXME 'u_preset': None,
                          'midi_notes': torch.zeros((latent_cell_input_shape[0], self._audio_seq_len, 2))}
            with torch.no_grad():
                summaries['latent_cell_{}'.format(i)] = torchinfo.summary(
                    latent_cell, input_data=input_data,
                    depth=5, verbose=0, device=torch.device('cpu'),
                    col_names=self._summary_col_names, row_settings=("depth", "var_names")
                )
        return summaries

    def _get_cells_output_shapes(self):
        x = torch.zeros(self._single_ch_input_size)
        output_sizes = list()
        with torch.no_grad():
            for cell in self.single_ch_cells:
                x = cell(x)
                output_sizes.append(x.shape)
        return output_sizes


class ConvLatentCell(nn.Module):
    def __init__(self, num_audio_input_ch: int, audio_input_H_W, audio_sequence_len: int,
                 num_latent_ch: int, num_layers: int,
                 kernel_size: Tuple[int, int], padding: Tuple[int, int], arch_args: Dict[str, bool]):
        """
        A class purely based on convolutions to mix various image feature maps into a latent feature map with
        a different number of channels.

        :param num_audio_input_ch: Number of channels in feature map extracted from a single audio element
            (e.g. a single MIDI Note)
        :param audio_input_H_W: Height and Width of audio feature maps that will be provided to this module
        :param audio_sequence_len:
        :param num_latent_ch:
        :param num_layers:
        """
        super().__init__()
        # FIXME ALWAYS add channels to mix preset values (use zeros if not used)
        total_num_input_ch = audio_sequence_len * num_audio_input_ch
        self.conv = nn.Sequential()
        # Self-attention for "not-that-small" feature maps only, i.e., feature maps which are larger
        #    than the 4x4 conv kernels from the main convolutional path
        if arch_args['att']:
            if np.prod(audio_input_H_W) > 16:
                self.conv.add_module('resatt', SelfAttentionConv2D(
                    total_num_input_ch, position_encoding=arch_args['posenc'], input_H_W=audio_input_H_W))
        else:
            if arch_args['posenc']:
                warnings.warn("'posenc' arch arg works with 'att' only (which is False), thus will be ignored.")
        num_out_ch = num_latent_ch if not arch_args['gated'] else 2 * num_latent_ch
        if num_layers == 1:
            self.conv.add_module('c', nn.Conv2d(total_num_input_ch, num_out_ch, kernel_size, 1, padding))
        elif num_layers == 2:  # No batch-norm inside the latent conv arch
            n_intermediate_ch = int(round(np.sqrt(total_num_input_ch * num_out_ch)))  # Geometric mean
            self.conv.add_module('c1', nn.Conv2d(total_num_input_ch, n_intermediate_ch, kernel_size, 1, padding))
            self.conv.add_module('act', nn.ELU())
            self.conv.add_module('c2', nn.Conv2d(n_intermediate_ch, num_out_ch, kernel_size, 1, padding))
        else:
            raise ValueError("Convolutional arch. for latent vector computation must contain <= 2 layers.")
        if arch_args['gated']:
            self.conv.add_module('gated', nn.GLU(dim=1))

    def forward(self, x_audio: torch.tensor,
                u_preset: Optional[torch.Tensor] = None, midi_notes: Optional[torch.Tensor] = None):
        """

        :param x_audio: Tensor with shape N x T x C x W x H where T is the original audio sequence length
        :param u_preset:
        :param midi_notes: May be used (or not, depends on the exact configuration) to add some "positional
            encoding" information to each sequence item in x_audio
        :return:
        """
        if u_preset is not None:
            return NotImplementedError()
        x_audio = torch.flatten(x_audio, start_dim=1, end_dim=2)  # merge sequence and channels dimensions
        return self.conv(x_audio)


class ConvLSTMLatentCell(nn.Module):
    def __init__(self, num_audio_input_ch: int, audio_input_H_W, audio_sequence_len: int,
                 num_latent_ch: int, num_layers: int,
                 kernel_size: Tuple[int, int], padding: Tuple[int, int], arch_args: Dict[str, bool]):
        """
        Latent cell based on a Convolutional LSTM (keeps the spatial structure of data)

        See ConvLatentCell to get information about parameters.
        """
        super().__init__()
        if arch_args['att']:
            warnings.warn("Self-attention arch arg 'att' is not valid and will be ignored.")
        # Main LSTM
        n_intermediate_ch = int(round(np.sqrt(num_audio_input_ch * num_latent_ch)))  # Geometric mean
        self.lstm_net = ConvLSTM(num_audio_input_ch, n_intermediate_ch, kernel_size, num_layers, bidirectional=True)
        # Positional encodings are added as residuals to help the LSTM to know which MIDI note is being processed.
        if arch_args['posenc']:
            self.note_encoder = nn.Sequential(
                nn.Linear(2, 4), nn.ReLU(),
                nn.Linear(4, np.prod(audio_input_H_W) * num_audio_input_ch), nn.Tanh()
            )
        else:
            self.note_encoder = None
        # Final regular conv to extract latent values, because the LSTM hidden state is activated (and these
        #    final activations might be undesired in the VAE framework). Also required because the bidirectional
        #    LSTM increases 2x the number of hidden channels.
        self.final_conv = nn.Conv2d(n_intermediate_ch*2, num_latent_ch, kernel_size, 1, padding)

    def forward(self, x_audio: torch.tensor,
                u_preset: Optional[torch.Tensor] = None, midi_notes: Optional[torch.Tensor] = None):
        if u_preset is not None:
            return NotImplementedError()  # FIXME also use preset feature maps (first hidden features?)
        # Learned residual positional encoding (per-pixel bias on the full feature maps)
        if midi_notes is not None and self.note_encoder is not None:
            for seq_idx in range(midi_notes.shape[1]):
                pos_enc = self.note_encoder(-0.5 + midi_notes[:, seq_idx, :] / 64.0)
                pos_enc = pos_enc.view(-1, *x_audio.shape[2:])
                if len(x_audio.shape) == 4:  # 4d tensor: raw audio input
                    x_audio[:, seq_idx, :, :] += pos_enc
                elif len(x_audio.shape) == 5:  # 5d tensor: spectrogram input
                    x_audio[:, seq_idx, :, :, :] += pos_enc
                else:
                    raise AssertionError()
        # Warning: the first n_intermediate_ch contains information about the preset itself and the first
        #    spectrogram only. This provides an "almost direct" path from preset to latent values (which we
        #    might want to avoid...???)
        # OTHER WARNING: h is a product of a sigmoid-activated and a tanh-activated value
        #    this implies a form a regularization before computing latent distributions' parameters
        output_sequence, last_states_per_layer = self.lstm_net(x_audio)
        return self.final_conv(output_sequence[:, -1, :, :, :])  # Use the last hidden state
