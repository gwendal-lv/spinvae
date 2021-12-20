
import warnings
from typing import Tuple, Dict, List, Optional
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn

from model.convlayer import Conv2D, TConv2D, ResBlock3Layers
from model import convlayer
from model import encoder  # Contains an architecture parsing method


class SpectrogramDecoder(nn.Module):
    """ Contains a spectrogram-input CNN and some MLP layers, and outputs the mu/logsigma2 values"""
    def __init__(self, architecture: str, dim_z: int, output_tensor_size: Tuple[int, int, int, int],
                 fc_dropout: float,
                 midi_notes: Optional[Tuple[Tuple[int, int]]] = None,
                 force_bigger_network=False):
        """

        :param architecture:
        :param dim_z:
        :param output_tensor_size:
        :param fc_dropout:
        :param force_bigger_network: If True, some layers will contain more channels. Optional, should be
            used to perform fair comparisons (same number of params) between different models.
        """
        super().__init__()
        self.output_tensor_size = output_tensor_size
        # Encoder input size is desired output size for this decoder (crop if too big? not necessary at the moment)
        self.spectrogram_input_size = (self.output_tensor_size[2], self.output_tensor_size[3])
        self.spectrogram_channels = output_tensor_size[1]
        self.dim_z = dim_z  # Latent-vector size
        self.midi_notes = midi_notes
        self.dim_w_single_ch_cnn = 512  # TODO contact MIDI notes to the style vector?  FIXME constant size
        if self.midi_notes is not None:
            self.dim_w_single_ch_cnn += 2  # midi pitch/vel will be concatenated to the style vector
        elif self.spectrogram_channels > 1:
            warnings.warn("midi_notes argument was not given but multi-channel spectrograms are being decoded.")
        self.full_architecture = architecture
        self.cnn_input_shape = None  # shape not including batch size

        self.base_arch_name, self.num_cnn_layers, self.num_fc_layers, self.arch_args \
            = encoder.parse_architecture(self.full_architecture)
        if self.base_arch_name == 'speccnn8l1':
            self.mixer_1x1conv_ch = 1024
            self.first_gt1_conv_ch = (512 if not force_bigger_network else 1800)
            self.cnn_input_shape = (self.mixer_1x1conv_ch, 3, 3)  # 16kHz sr: 3x3 instead of 3x4
        elif self.base_arch_name.startswith('sprescnn'):
            self.mixer_1x1conv_ch = 96 if self.arch_args['time+'] else 192  # Much larger feature maps?
            self.first_gt1_conv_ch = 1024
            if force_bigger_network:
                raise NotImplementedError()
            # Encoder output size: 5, 16 if better_time_res else 5, 4
            self.cnn_input_shape = (self.mixer_1x1conv_ch, 4, (16 if self.arch_args['time+'] else 4))
        else:
            raise NotImplementedError()
        self.fc_dropout = fc_dropout

        # - - - - - 1) MLP output size must to correspond to encoder's MLP input size - - - -
        self.mlp = nn.Sequential()
        for i in range(self.num_fc_layers):
            in_units = self.dim_z if (i == 0) else 1024
            out_units = 1024 if (i < (self.num_fc_layers - 1)) else int(np.prod(self.cnn_input_shape))
            self.mlp.add_module('decfc{}'.format(i), nn.Linear(in_units, out_units))
            # No final ReLU (and leads to worse generalization, but don't know why)
            if i < (self.num_fc_layers - 1):
                self.mlp.add_module('decact{}'.format(i), nn.ReLU())
            # TODO try remove this dropout? or dropout before?
            if self.fc_dropout > 0.0:
                self.mlp.add_module("decdrop{}".format(i), nn.Dropout(self.fc_dropout))

        # - - - - - - - - - - 2) Features "un-mixer" - - - - - - - - - -
        self.features_unmixer_cnn = Conv2D(self.mixer_1x1conv_ch, self.spectrogram_channels * self.first_gt1_conv_ch,
                                           [1, 1], [1, 1], 0, act=nn.LeakyReLU(0.1),
                                           )  # TODO adaIN  with all MIDI notes given to that layer?
        # - - - - - and 3) Main CNN decoder (applied once per spectrogram channel) - - - - -
        single_spec_output_size = list(self.output_tensor_size)
        single_spec_output_size[1] = 1  # Single-channel output
        self.single_ch_cnn = nn.Sequential()

        if self.base_arch_name == 'speccnn8l1':
            ''' Inspired by the wavenet baseline spectral autoencoder, but all sizes are drastically reduced
            (especially the last, not so useful but very GPU-expensive layer on large images) '''
            _in_ch, _out_ch = -1, -1  # backup for res blocks
            for i in range(1, self.num_cnn_layers):  # 'normal' channels: (1024, ) 512, 256, ...., 8
                in_ch = self.first_gt1_conv_ch if i == 1 else 2 ** (10 - i)
                out_ch = 2 ** (10 - (i + 1)) if (i < (self.num_cnn_layers - 1)) else 1
                if self.arch_args['bigger']:
                    in_ch = max(in_ch, 128)
                    if out_ch > 1:
                        out_ch = max(out_ch, 128)
                elif self.arch_args['big']:
                    in_ch = in_ch if in_ch > 64 else in_ch * 2
                    if out_ch > 1:
                        out_ch = out_ch if out_ch > 64 else out_ch * 2
                padding, stride = (2, 2), 2
                if i < self.num_cnn_layers - 1:
                    kernel = (4, 4)
                else:
                    kernel = (5, 5)
                if i in [1, 2, 3, 4]:
                    output_padding = (1, 1)
                elif i in [5, 6]:
                    output_padding = (1, 0)
                else:  # last layer
                    output_padding = (0, 0)
                if 0 <= i < (self.num_cnn_layers-1):
                    act = nn.LeakyReLU(0.1)
                    # keep using BN with some layers (don't use AdaIN only, BN is very effective)
                    if 1 <= i <= 4:
                        norm = 'bn+adain' if self.arch_args['adain'] else 'bn'
                    elif i == 5:  # TODO try adain on layer 6?
                        norm = 'adain' if self.arch_args['adain'] else 'bn'
                    else:
                        norm = 'bn'
                else:
                    norm = None
                    act = nn.Identity()
                if self.arch_args['res'] and i in [1, 3]:
                    _in_ch, _out_ch = in_ch, out_ch
                elif self.arch_args['res'] and i in [2, 4]:
                    l = convlayer.ResTConv2D(_in_ch, in_ch, out_ch, kernel, stride, padding, output_padding, act=act,
                                             norm_layer=norm, adain_num_style_features=self.dim_w_single_ch_cnn)
                    self.single_ch_cnn.add_module('dec_{}_{}'.format(i-1, i), l)
                else:
                    l = convlayer.TConv2D(in_ch, out_ch, kernel, stride, padding, output_padding, act=act,
                                          norm_layer=norm, adain_num_style_features=self.dim_w_single_ch_cnn)
                    self.single_ch_cnn.add_module('dec{}'.format(i), l)

        elif self.base_arch_name.startswith('sprescnn'):
            better_time_res = (self.base_arch_name == 'sprescnnt')
            norm = 'bn+adain' if self.arch_args['adain'] else 'bn'
            # See encoder.py: this decoder is quite symmetrical (not really...) to the corresponding encoder
            # However, the "features mixer" layer is always the first
            main_blocks_indices = [0, 1, 2, 3, 4, 5]
            if self.arch_args['big']:
                res_blocks_counts = [3, 4, 3, 2, 1, 1]
                res_blocks_ch = [self.first_gt1_conv_ch, 512, 256, 128, 128, 64]
            else:
                res_blocks_counts = [2, 3, 2, 2, 1, 1]
                res_blocks_ch = [self.first_gt1_conv_ch, 512, 256, 128, 32, 8]
            self.num_cnn_layers = sum(res_blocks_counts)
            layer_idx = -1
            for main_block_idx in main_blocks_indices:
                for res_block_idx in range(res_blocks_counts[main_block_idx]):
                    layer_idx += 1
                    if layer_idx < (self.num_cnn_layers - 1):  # All layers but the last
                        in_ch = res_blocks_ch[main_block_idx]
                        # Last block of a main block adapts the size for the following block
                        if res_block_idx == (res_blocks_counts[main_block_idx] - 1):
                            out_ch = res_blocks_ch[main_block_idx + 1]
                            if main_block_idx <= 1 and self.arch_args['time+']:
                                upsample = (True, False)  # Optional: No temporal stride for first layers
                            else:
                                upsample = (True, True)
                        else:
                            out_ch = res_blocks_ch[main_block_idx]
                            upsample = (False, False)
                        if main_block_idx in [3] and res_block_idx == 1:
                            extra_padding = (1, 0)  # We need some output padding to get a larger frequency axis
                        else:
                            extra_padding = (0, 0)
                        l = ResBlock3Layers(in_ch, in_ch//4, out_ch, act=nn.LeakyReLU(0.1),
                                            upsample=upsample, extra_padding=extra_padding,
                                            norm_layer=norm, adain_num_style_features=self.dim_w_single_ch_cnn)
                        self.single_ch_cnn.add_module('resblk{}'.format(layer_idx), l)
                    else:  # Last layer: no norm or act, 6x6 Tconv (https://distill.pub/2016/deconv-checkerboard/)
                        kernel_size, padding = ((6, 6), (2, 2)) if self.arch_args['big'] else ((4, 4), (1, 1))
                        self.single_ch_cnn.add_module('tconv',
                                                      TConv2D(res_blocks_ch[-1], 1, kernel_size, (2, 2),
                                                              padding=padding, output_padding=(0, 0),
                                                              act=nn.Identity(), norm_layer=None))

        else:
            raise NotImplementedError()

        # Final activation, for all architectures
        self.single_ch_cnn_act = nn.Hardtanh()

    def forward(self, z_sampled, w_style=None):
        if w_style is None:  # w_style can be omitted during summary writing
            print("[decoder.py] w_style tensor is None; replaced by 0.0 (should happen during init only)")
            w_style = torch.zeros((z_sampled.shape[0], self.dim_w_single_ch_cnn), device=z_sampled.device)
        mixed_features = self.mlp(z_sampled)
        mixed_features = mixed_features.view(-1,  # batch size auto inferred
                                             self.cnn_input_shape[0], self.cnn_input_shape[1], self.cnn_input_shape[2])
        unmixed_features, _ = self.features_unmixer_cnn((mixed_features, None))  # No style given to feats mixer
        # This won't work on multi-channel spectrograms with force_bigger_network==True (don't do this anyway)
        single_ch_cnn_inputs = torch.split(unmixed_features, self.first_gt1_conv_ch,  # actual multi-spec: 512ch-split
                                           dim=1)  # Split along channels dimension
        single_ch_cnn_outputs = list()
        for i, single_ch_in in enumerate(single_ch_cnn_inputs):
            # concat midi notes to w_style (if w_style was not None as a method argument)
            if self.midi_notes is not None and w_style.shape[1] < self.dim_w_single_ch_cnn:
                notes_tensor = torch.tensor(self.midi_notes[i], device=w_style.device, dtype=w_style.dtype)
                notes_tensor = notes_tensor / 63.5 - 1.0
                w_with_midi = torch.cat([notes_tensor.expand(w_style.shape[0], 2), w_style], dim=1)
            else:
                w_with_midi = w_style
            single_ch_cnn_outputs.append(self.single_ch_cnn((single_ch_in, w_with_midi)))
        # Remove style
        single_ch_cnn_outputs = [x[0] for x in single_ch_cnn_outputs]

        # Concatenate single-ch spectrograms, crop if necessary
        single_ch_cnn_outputs = torch.cat(single_ch_cnn_outputs, dim=1)
        if self.base_arch_name.startswith('sprescnn'):  # Output is 264 x 256 ; target is 257 x 251 (+ 4.7% pixels)
            single_ch_cnn_outputs = single_ch_cnn_outputs[:, :, 3:3+257, 2:2+251]

        # Final activation
        return self.single_ch_cnn_act(single_ch_cnn_outputs)

    def get_fc_layers_parameters(self) -> Dict[str, Dict[str, np.ndarray]]:
        """ Returns a dict of dicts: weights and biases of all FC layers at the decoder's input.
        1st dict keys are layers names, 2nd dict keys are 'weight' and 'bias'. To be used for Tensorboard hists.

        Clones are returned (can be used in a different thread - don't need another copy). """
        layers_params = OrderedDict()
        fc_index = 0
        for layer in self.mlp:
            if isinstance(layer, nn.Linear):
                w = layer.weight.clone().detach().cpu().numpy().flatten()
                layers_params['FC{}'.format(fc_index)] \
                    = {'weight': w, 'weight_abs': np.abs(w),
                       'bias': layer.bias.clone().detach().cpu().numpy().flatten()}
                fc_index += 1
        return layers_params



if __name__ == "__main__":

    import torchinfo
    output_size = (160, 1, 257, 251)
    dim_z = 200
    dec = SpectrogramDecoder('speccnn8l1', dim_z, output_size, 0.1)  # sprescnn_time+
    p = dec.get_fc_layers_parameters()

    #out_test = dec(torch.zeros((160, dim_z)))
    #_ = torchinfo.summary(dec, input_size=(160, dim_z), depth=3)

    # test plot layers weights (matplotlib, not tensorboard)
    from utils import figures
    import matplotlib.pyplot as plt
    figures.plot_network_parameters(p)
    plt.show()

