import warnings
from typing import Dict, Tuple, List, Sequence, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchinfo

import utils.probability
from model.ladderbase import LadderBase
from model.presetmodel import PresetDecoder
from model.convlayer import ConvBlock2D, UpsamplingResBlock, SelfAttentionConv2D, ResBlockBase
from data.preset2d import Preset2dHelper


class LadderDecoder(LadderBase):
    def __init__(self, conv_arch, latent_arch,
                 latent_tensors_shapes: List[Sequence[int]], audio_output_shape: Sequence[int],
                 audio_proba_distribution: str,
                 preset_architecture: Optional[str] = None,
                 preset_hidden_size: Optional[int] = None,
                 preset_numerical_proba_distribution: Optional[str] = None,
                 preset_helper: Optional[Preset2dHelper] = None):
        # TODO doc
        super().__init__(conv_arch, latent_arch)
        self.n_latent_levels = len(latent_tensors_shapes)
        self._latent_tensors_shapes = latent_tensors_shapes
        self._audio_output_shape = audio_output_shape
        self.num_audio_output_ch = self._audio_output_shape[1]
        self._cells_input_shapes = list()

        # - - - - - 0) Output probability distribution helper class - - - - -
        if audio_proba_distribution.lower() == "gaussian_unitvariance":
            self.audio_proba_distribution = utils.probability.GaussianUnitVariance()
        else:
            raise ValueError("Unavailable audio probability distribution {}".format(audio_proba_distribution))

        # - - - - - 1) Build the single-channel CNN (outputs a single reconstructed audio channel) - - - - -
        # We build these cells first even if it will be applied after the latent cells.
        # The first cell takes only latent values as input, and other cells will use both previous features and
        # a new level of latent values
        conv_args = self.single_ch_conv_arch['args']
        n_blocks = self.single_ch_conv_arch['n_blocks']
        if conv_args['depsep5x5'] and self.single_ch_conv_arch['n_layers_per_block'] == 1:
            raise AssertionError("Depth-separable convolutions require at least 2 layers per res-block.")

        if self.single_ch_conv_arch['name'].startswith('specladder'):
            if conv_args['adain']:
                raise NotImplementedError()
            if conv_args['att'] and self.single_ch_conv_arch['n_layers_per_block'] < 2:
                raise ValueError("'_att' conv arg will add a residual self-attention layer and requires >= 2 layers")
            if conv_args['att'] and conv_args['depsep5x5'] and self.single_ch_conv_arch['n_layers_per_block'] < 3:
                raise ValueError("'_att' and '_depsep5x5' conv args (both provided) require >= 3 layers")
            self.single_ch_cells = list()
            if 1 <= self.n_latent_levels <= (n_blocks - 2):
                cells_first_block = list(range(0, self.n_latent_levels))
            else:
                raise NotImplementedError("Cannot build encoder with {} latent levels".format(self.n_latent_levels))

            for i_blk in range(n_blocks):
                residuals_path = nn.Sequential()
                blk_in_ch = 2**(10 - i_blk)  # number of input channels
                blk_out_ch = 2**(9 - i_blk)  # number of ch decreases after each strided Tconv (at block's end)
                if conv_args['big']:
                    blk_in_ch, blk_out_ch = blk_in_ch * 2, blk_out_ch * 2
                min_ch = 1 if not conv_args['bigger'] else 128
                max_ch = 512 if not conv_args['bigger'] else 1024
                blk_in_ch, blk_out_ch = np.clip([blk_in_ch, blk_out_ch], min_ch, max_ch)
                blk_hid_ch = blk_out_ch  # base number of internal (hidden) block channels
                is_last_block = (i_blk == (n_blocks - 1))
                if is_last_block:
                    blk_out_ch = self.audio_proba_distribution.num_parameters

                # First block of cell? Create new cell, store number of input channels
                if i_blk in cells_first_block:
                    self.single_ch_cells.append(nn.Sequential())
                    self._cells_input_shapes.append([audio_output_shape[0], blk_in_ch, -1, -1])

                if not is_last_block:  # Hidden res blocks can contain multiple conv blocks
                    # Output padding (1, 1): with 4x4 kernels in hidden layers, this ensures same hidden feature maps
                    #  HxW in the encoder and this decoder. Output will be a bit too big but can be cropped.
                    conv = nn.ConvTranspose2d(blk_in_ch, blk_hid_ch, (4, 4), 2, 2, (1, 1))
                    residuals_path.add_module('stridedT', ConvBlock2D(
                        conv, self._get_conv_act(), self._get_conv_norm(blk_in_ch), 'nac'))
                    # Usual convs are applied after upsampling (less parameters)
                    for j in range(self.single_ch_conv_arch['n_layers_per_block'] - 1):
                        if j == 0 and conv_args['att'] and (2 <= i_blk <= 4):  # attention costs a lot of GPU RAM
                            self_att = SelfAttentionConv2D(blk_hid_ch, position_encoding=True)
                            residuals_path.add_module('self_att', ConvBlock2D(
                                self_att, None, self._get_conv_norm(blk_hid_ch), 'nc'))
                        elif conv_args['depsep5x5']:  # Structure close to NVAE (NeurIPS 2020)
                            depth_sep_ch = blk_hid_ch * 2  # 3x channels expansion costs a lot of GPU RAM
                            sub_conv_block = nn.Sequential()
                            conv = nn.Conv2d(blk_hid_ch, depth_sep_ch, (1, 1))
                            sub_conv_block.add_module('more_ch_' + str(j), ConvBlock2D(
                                conv, None, self._get_conv_norm(blk_hid_ch), 'nc'))
                            conv = nn.Conv2d(depth_sep_ch, depth_sep_ch, (5, 5), 1, 2, groups=depth_sep_ch)
                            sub_conv_block.add_module('depthsep_' + str(j), ConvBlock2D(
                                conv, self._get_conv_act(), self._get_conv_norm(depth_sep_ch), 'nac'))
                            conv = nn.Conv2d(depth_sep_ch, blk_hid_ch, (1, 1))
                            sub_conv_block.add_module('less_ch_' + str(j), ConvBlock2D(
                                conv, self._get_conv_act(), self._get_conv_norm(depth_sep_ch), 'nac'))
                            # As this module contains 3 convs, we use it as residuals not to impair gradient backprop
                            residuals_path.add_module('depsep5x5_' + str(j), ResBlockBase(sub_conv_block))
                        else:
                            conv = nn.Conv2d(blk_hid_ch, blk_hid_ch, (3, 3), 1, 1)
                            residuals_path.add_module('conv', ConvBlock2D(
                                conv, self._get_conv_act(), self._get_conv_norm(blk_hid_ch), 'nac'))
                else:  # last block: conv only, wider 5x5 kernel
                    residuals_path.add_module('stridedT', ConvBlock2D(
                        nn.ConvTranspose2d(blk_in_ch, blk_out_ch, (5, 5), 2, 2, 0), None, None, 'c'))

                if conv_args['res'] and not is_last_block:
                    current_block = UpsamplingResBlock(residuals_path)
                else:
                    current_block = residuals_path  # No skip-connection
                self.single_ch_cells[-1].add_module('blk{}'.format(i_blk), current_block)

        else:  # Spectrograms only are supported at the moment
            raise NotImplementedError(
                "Unimplemented '{}' architecture".format(self.single_ch_conv_arch['name']))

        # - - - - - 2) Build latent residual networks - - - - -
        # To refine main conv feature maps using the different levels of latent values
        # Warning: (main convolutional) cells and latent cells ordering is opposite in the decoder
        self.latent_cells = list()

        # The decoder ctor is different from the encoder, because the only latent architecture it provides
        # is the feedforward convolutions (in latent cells). So, those cells are built directly inside this
        # ctor instead of being built in a separate class. (might be refactored in the future)
        if self.latent_arch['name'] == "conv" or self.latent_arch['name'] == 'lstm':  # FIXME
            if self.latent_arch['name'] == 'lstm':
                warnings.warn("Decoder: LSTM latent structure not implemented (the usual conv will be used instead)")
            # Convolutional latent structure: increase number of channels of latent features maps. The new channels
            # will be used as residuals added to the main convolutional path.
            if self.latent_arch['args']['k1x1']:
                kernel_size, padding = (1, 1), (0, 0)
            elif self.latent_arch['args']['k3x3']:
                kernel_size, padding = (3, 3), (1, 1)
            else:
                raise NotImplementedError("Can't build latent cells: conv kernel arg ('_k1x1' or '_k3x3') not provided")
            for latent_level, latent_tensor_shape in enumerate(self._latent_tensors_shapes):
                cell_index = self._get_cell_index(latent_level)
                self._cells_input_shapes[cell_index][2:] = latent_tensors_shapes[latent_level][2:]
                n_latent_input_ch = self._latent_tensors_shapes[latent_level][1]
                n_latent_output_ch = self._cells_input_shapes[cell_index][1] * self.num_audio_output_ch
                cell = nn.Sequential()
                if self.latent_arch['args']['att'] and n_latent_input_ch >= 8:  # Useless with a small num. of channels
                    input_H_W = tuple(self._cells_input_shapes[cell_index][2:])
                    if np.prod(input_H_W) > 16:
                        cell.add_module('resatt', SelfAttentionConv2D(
                            n_latent_input_ch, n_latent_input_ch,
                            position_encoding=self.latent_arch['args']['posenc'], input_H_W=input_H_W)
                        )
                else:
                    if self.latent_arch['args']['posenc']:
                        warnings.warn("'posenc' arch arg works with 'att' only (which is False), thus will be ignored.")
                if self.latent_arch['args']['gated']:
                    n_latent_output_ch *= 2  # GLU will half the number of outputs
                if self.latent_arch['n_layers'] == 1:
                    cell.add_module('c', nn.Conv2d(n_latent_input_ch, n_latent_output_ch, kernel_size, 1, padding))
                elif self.latent_arch['n_layers'] == 2:  # No batch-norm inside the latent conv arch
                    n_intermediate_ch = int(round(np.sqrt(n_latent_input_ch * n_latent_output_ch)))
                    cell.add_module('c1', nn.Conv2d(n_latent_input_ch, n_intermediate_ch, kernel_size, 1, padding))
                    cell.add_module('act', nn.ELU())
                    cell.add_module('c2', nn.Conv2d(n_intermediate_ch, n_latent_output_ch, kernel_size, 1, padding))
                else:
                    raise ValueError("Convolutional arch. for latent vector computation must contain <= 2 layers.")
                if self.latent_arch['args']['gated']:
                    cell.add_module('gated', nn.GLU(dim=1))
                self.latent_cells.append(cell)

        else:
            raise NotImplementedError("Cannot build latent arch {}: name not implemented".format(latent_arch))

        # 3) Preset decoder (separate class)
        if preset_architecture is not None:
            assert (preset_numerical_proba_distribution is not None and preset_helper is not None
                    and preset_hidden_size is not None)
            self.preset_decoder = PresetDecoder(
                preset_architecture,
                self._latent_tensors_shapes,
                preset_hidden_size,
                preset_numerical_proba_distribution,
                preset_helper
            )
        else:
            self.preset_decoder = None

        # Finally, Python lists must be converted to nn.ModuleList to be properly recognized by PyTorch
        self.single_ch_cells = nn.ModuleList(self.single_ch_cells)
        self.latent_cells = nn.ModuleList(self.latent_cells)

    def get_custom_group_module(self, group_name: str):
        """ Returns a module """
        if group_name == 'audio':
            return self.single_ch_cells
        elif group_name == 'latent':
            return self.latent_cells
        elif group_name == 'preset':
            return self.preset_decoder  # Can be None
        else:
            raise ValueError("Unavailable group_name '{}'".format(group_name))

    def forward(self, z_sampled: List[torch.Tensor], u_target: Optional[torch.Tensor] = None):
        """ Returns the p(x|z) probability distributions and values sampled from them,
            and preset_decoder_out (tuple of 5 values) if this instance has a preset decoder, and
                if u_target is not None. """
        # ---------- Audio decoder ----------
        # apply latent cells and split outputs to get residuals
        conv_cell_res_inputs = [[] for _ in range(self.n_latent_levels)]  # 1st dim: cell index; 2nd dim: audio channel
        for latent_level, level_z in enumerate(z_sampled):  # Higher latent_level corresponds to deeper latent features
            cell_index = self._get_cell_index(latent_level)
            multi_ch_conv_input = self.latent_cells[latent_level](level_z)
            # TODO FIXME maybe don't chunk
            conv_cell_res_inputs[cell_index] = torch.chunk(multi_ch_conv_input, self.num_audio_output_ch, 1)
        # Sequential CNN audio decoder - apply cell-by-cell
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

        # ---------- Preset decoder ----------
        if self.preset_decoder is not None and u_target is not None:
            # TODO teacher-forcing or not?
            # TODO NEVER use teaching forcing during eval
            preset_decoder_out = self.preset_decoder(z_sampled, u_target)
        else:
            preset_decoder_out = (None, ) * 5

        # All audio outputs are concatenated (channels dimension) to remain usable in a multi-GPU configuration
        return torch.cat(audio_prob_parameters, dim=1), torch.cat(audio_x_sampled, dim=1), preset_decoder_out

    def audio_log_prob_loss(self, audio_prob_parameters, x_in):
        return self.audio_proba_distribution.NLL(audio_prob_parameters, x_in)

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
                depth=7, verbose=0, device=torch.device('cpu'),
                col_names=self._summary_col_names, row_settings=("depth", "var_names")
            )

    def get_latent_cells_summaries(self):
        summaries = dict()
        for latent_level, latent_cell in enumerate(self.latent_cells):
            input_size = list(self._latent_tensors_shapes[latent_level])
            input_size[0] = 1  # Single-element minibatch
            with torch.no_grad():
                summaries['latent_cell_{}'.format(latent_level)] = torchinfo.summary(
                    latent_cell, input_size=input_size,
                    depth=5, verbose=0, device=torch.device('cpu'),
                    col_names=self._summary_col_names, row_settings=("depth", "var_names")
                )
        return summaries
