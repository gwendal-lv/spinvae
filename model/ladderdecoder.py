from typing import Dict, Tuple, List, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchinfo

import utils.probability
from model.convlayer import ConvBlock2D


class LadderDecoder(nn.Module):

    def __init__(self, conv_arch, latent_arch,
                 latent_tensors_shapes: List[Sequence[int]], audio_output_shape: Sequence[int],
                 audio_proba_distribution: str):
        # TODO doc
        super().__init__()
        self.single_ch_conv_arch = conv_arch
        self.latent_arch = latent_arch
        self.n_latent_levels = len(latent_tensors_shapes)
        self._latent_tensors_shapes = latent_tensors_shapes
        self._audio_output_shape = audio_output_shape
        self.num_audio_output_ch = self._audio_output_shape[1]
        self._cells_input_shapes = list()

        # - - - - - 1) Build the single-channel CNN (outputs a single reconstructed audio channel) - - - - -
        # We build these cells first even if it will be applied after the latent cells.
        # The first cell takes only latent values as input, and other cells will use both previous features and
        # a new level of latent values
        conv_args = self.single_ch_conv_arch['args']
        n_blocks = self.single_ch_conv_arch['n_blocks']
        if self.single_ch_conv_arch['name'].startswith('specladder'):
            if conv_args['big'] or conv_args['bigger'] or conv_args['adain'] or conv_args['res'] or conv_args['att']:
                raise NotImplementedError()
            self.single_ch_cells = list()
            if self.n_latent_levels == 1:
                cells_first_block = [0]
            elif self.n_latent_levels == 2:
                cells_first_block = [0, n_blocks // 2]
            else:
                raise NotImplementedError("Cannot build encoder with {} latent levels".format(self.n_latent_levels))

            for i_blk in range(n_blocks):
                current_block = nn.Sequential()
                blk_in_ch = 2**(10 - i_blk)  # number of input channels
                blk_hid_ch = blk_in_ch  # base number of internal (hidden) block channels
                blk_out_ch = 2**(9 - i_blk)  # number of ch decreases after each strided Tconv (at block's end)
                blk_in_ch, blk_hid_ch, blk_out_ch = min(blk_in_ch, 512), min(blk_hid_ch, 512), min(blk_out_ch, 512)
                is_last_block = (i_blk == (n_blocks - 1))
                if is_last_block:
                    blk_out_ch = 1  # FIXME depends on the probability distribution
                    kernel_size = (5, 5)
                else:
                    kernel_size = (4, 4)
                # Output padding: with 4x4 kernels in hidden layers, this ensures same hidden feature maps HxW in the
                # encoder and this decoder. Output will be a bit too big but can be cropped
                output_pad = (1, 1) if not is_last_block else (0, 0)

                # First block of cell? Create new cell, store number of input channels
                if i_blk in cells_first_block:
                    self.single_ch_cells.append(nn.Sequential())
                    self._cells_input_shapes.append([audio_output_shape[0], blk_in_ch, -1, -1])

                n_layers = self.single_ch_conv_arch['n_layers_per_block'] if not is_last_block else 1
                if n_layers >= 2:
                    raise NotImplementedError()
                if n_layers >= 1:
                    conv = nn.ConvTranspose2d(blk_hid_ch, blk_out_ch, kernel_size, (2, 2), 2, output_pad)
                    act = nn.LeakyReLU(0.1) if not is_last_block else None
                    norm = nn.BatchNorm2d(blk_hid_ch) if not is_last_block else None
                    current_block.add_module('stridedT', ConvBlock2D(conv, act, norm, 'nac'))

                self.single_ch_cells[-1].add_module('blk{}'.format(i_blk), current_block)

        else:  # Spectrograms only are supported at the moment
            raise NotImplementedError(
                "Unimplemented '{}' architecture".format(self.single_ch_conv_arch['name']))

        # - - - - - 2) Build latent residual networks - - - - -
        # To refine main conv feature maps using the different levels of latent values
        # Warning: (main convolutional) cells and latent cells ordering is opposite in the decoder
        self.latent_cells = list()

        if self.latent_arch['name'].startswith("conv"):
            # Convolutional latent structure: increase number of channels of latent features maps. The new channels
            # will be used as residuals added to the main convolutional path.
            if self.latent_arch['name'] == 'convk11':
                kernel_size, padding = (1, 1), (0, 0)
            elif self.latent_arch['name'] == 'convk33':
                kernel_size, padding = (3, 3), (1, 1)
            else:
                raise NotImplementedError("Cannot build latent arch {}: name not implement".format(latent_arch))
            if self.latent_arch['n_layers'] != 1:  # FIXME we should allow 2-layer conv latent inference
                raise ValueError("Convolutional architecture for latent vector computation must be 1-layer.")
            for latent_level, latent_tensor_shape in enumerate(self._latent_tensors_shapes):
                cell_index = self._get_cell_index(latent_level)
                n_latent_output_ch = self._cells_input_shapes[cell_index][1] * self.num_audio_output_ch
                self.latent_cells.append(nn.Conv2d(
                    self._latent_tensors_shapes[latent_level][1], n_latent_output_ch, kernel_size, 1, padding
                ))
                # We can finish to set the (conv) cell input shape
                self._cells_input_shapes[cell_index][2:] = latent_tensors_shapes[latent_level][2:]

        else:
            raise NotImplementedError("Cannot build latent arch {}: name not implemented".format(latent_arch))

        # - - - - - 3) Probability distribution helper class - - - - -
        if audio_proba_distribution.lower() == "gaussian_unitvariance":
            self.audio_proba_distribution = utils.probability.GaussianUnitVariance()
        else:
            raise ValueError("Unavailable audio probability distribution {}".format(audio_proba_distribution))

        # Finally, Python lists must be converted to nn.ModuleList to be properly recognized by PyTorch
        self.single_ch_cells = nn.ModuleList(self.single_ch_cells)
        self.latent_cells = nn.ModuleList(self.latent_cells)


    def forward(self, z_sampled: List[torch.Tensor]):
        """ Returns the p(x|z) probability distributions and values sampled from them. """
        # apply latent cells and split outputs
        conv_cell_res_inputs = [[] for _ in range(self.n_latent_levels)]  # 1st dim: cell index; 2nd dim: audio channel
        for latent_level, level_z in enumerate(z_sampled): # Higher latent_level corresponds to deeper latent features
            cell_index = self._get_cell_index(latent_level)
            multi_ch_conv_input = self.latent_cells[latent_level](level_z)
            conv_cell_res_inputs[cell_index] = torch.chunk(multi_ch_conv_input, self.num_audio_output_ch, 1)

        # Sequential CNN decoder - apply cell-by-cell
        audio_prob_parameters = list()  # One probability distribution tensor / audio channel
        for audio_ch in range(self.num_audio_output_ch):
            x = torch.zeros_like(conv_cell_res_inputs[0][audio_ch])
            for cell_index, cell in enumerate(self.single_ch_cells):
                x += conv_cell_res_inputs[cell_index][audio_ch]
                x = cell(x)
            # crop and append to the list of per-channel outputs
            if self.single_ch_conv_arch['name'].startswith('specladder8x'):  # known CNN output size: 257x257
                audio_prob_parameters.append(x[:, :, :, 3:254])
            else:
                raise AssertionError("Cropping not implemented for the specific convolutional architecture '{}'."
                                     .format(self.single_ch_conv_arch['name']))
        # Apply activations
        audio_prob_parameters = [self.audio_proba_distribution.apply_activations(x) for x in audio_prob_parameters]
        # Sample from the probability distribution should always be fast and easy (even for mixture models)
        audio_x_sampled = [self.audio_proba_distribution.sample(x) for x in audio_prob_parameters]

        # All outputs are concatenated (channels dimension) to remain usable in a multi-GPU configuration
        return torch.cat(audio_prob_parameters, dim=1), torch.cat(audio_x_sampled, dim=1)

    def audio_log_prob_loss(self, audio_prob_parameters, x_in):
        return self.audio_proba_distribution.log_prob(audio_prob_parameters, x_in)

    def _get_cell_index(self, latent_level: int):
        """ Returns the convolutional cell index which corresponds to a given latent levels. The ordering is reversed
        in the decoder: deepest latent features (extract within the encoder) have the higher level indices. """
        return (self.n_latent_levels - 1) - latent_level

    @property
    def _summary_col_names(self):
        return "input_size", "kernel_size", "output_size", "num_params", "mult_adds"

    def get_single_ch_conv_summary(self):
        """ Torchinfo summary of the CNN that sequentially reconstructs each audio channel from latent values. """
        single_ch_cnn_without_latent = nn.Sequential()
        for i, cell in enumerate(self.single_ch_cells):
            single_ch_cnn_without_latent.add_module('cell{}'.format(i), cell)
        # FIXME this won't work for non-convolutional latent architectures (works with residuals)
        input_shape = list(self._latent_tensors_shapes[-1])  # Latents are used in the reverse order
        input_shape[0], input_shape[1] = 1, self._cells_input_shapes[0][1]  # After adaptation of number of channels
        with torch.no_grad():
            return torchinfo.summary(
                single_ch_cnn_without_latent, input_size=input_shape,
                depth=5, verbose=0, device=torch.device('cpu'),
                col_names=self._summary_col_names, row_settings=("depth", "var_names")
            )

    def get_latent_cells_summaries(self):
        summaries = dict()
        for latent_level, latent_cell in enumerate(self.latent_cells):
            with torch.no_grad():
                summaries['latent_cell_{}'.format(latent_level)] = torchinfo.summary(
                    latent_cell, input_size=self._latent_tensors_shapes[latent_level],
                    depth=5, verbose=0, device=torch.device('cpu'),
                    col_names=self._summary_col_names, row_settings=("depth", "var_names")
                )
        return summaries
