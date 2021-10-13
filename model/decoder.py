import warnings

import numpy as np
import torch
import torch.nn as nn
from typing import Tuple

from model import convlayer
from model import encoder  # Contains an architecture parsing method


class SpectrogramDecoder(nn.Module):
    """ Contains a spectrogram-input CNN and some MLP layers, and outputs the mu/logsigma2 values"""
    def __init__(self, architecture: str, dim_z: int, output_tensor_size: Tuple[int, int, int, int], fc_dropout: float,
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
        self.full_architecture = architecture
        self.cnn_input_shape = None  # shape not including batch size

        self.base_arch_name, self.num_cnn_layers, self.num_fc_layers, self.arch_args \
            = encoder.parse_architecture(self.full_architecture)

        self.mixer_1x1conv_ch = 1024  # Valid for all architectures
        self.last_4x4conv_ch = (512 if not force_bigger_network else 1800)
        self.fc_dropout = fc_dropout

        # - - - - - 1) MLP output size must to correspond to encoder's MLP input size - - - -
        assert self.base_arch_name.startswith('speccnn')
        self.cnn_input_shape = (self.mixer_1x1conv_ch, 3, 3)  # 16kHz sr: 3x3 instead of 3x4
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
        self.features_unmixer_cnn = convlayer.TConv2D(self.mixer_1x1conv_ch,
                                                      self.spectrogram_channels * self.last_4x4conv_ch,
                                                      [1, 1], [1, 1], 0,
                                                      act=nn.LeakyReLU(0.1),
                                                      )  # TODO adaIN  with all MIDI notes given to that layer
        # - - - - - and 3) Main CNN decoder (applied once per spectrogram channel) - - - - -
        single_spec_output_size = list(self.output_tensor_size)
        single_spec_output_size[1] = 1  # Single-channel output
        self.single_ch_cnn = nn.Sequential()
        if self.base_arch_name == 'speccnn8l1':
            ''' Inspired by the wavenet baseline spectral autoencoder, but all sizes are drastically reduced
            (especially the last, not so useful but very GPU-expensive layer on large images) '''
            _in_ch, _out_ch = -1, -1  # backup for res blocks
            for i in range(1, self.num_cnn_layers):  # 'normal' channels: (1024, ) 512, 256, ...., 8
                in_ch = self.last_4x4conv_ch if i == 1 else 2 ** (10 - i)
                out_ch = 2 ** (10 - (i + 1)) if (i < (self.num_cnn_layers - 1)) else 1
                if self.arch_args['big']:
                    in_ch = max(in_ch, 128)
                    if out_ch > 1:
                        out_ch = max(out_ch, 128)
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
                    norm = 'adain' if self.arch_args['adain'] else 'bn'
                else:
                    norm = None
                    act = nn.Identity()
                if self.arch_args['res'] and i in [1, 3]:
                    _in_ch, _out_ch = in_ch, out_ch
                elif self.arch_args['res'] and i in [2, 4]:
                    l = convlayer.ResTConv2D(_in_ch, in_ch, out_ch, kernel, stride, padding, output_padding, act=act,
                                             norm_layer=norm, adain_num_style_features=dim_z)  # TODO concat MIDI note ?
                    self.single_ch_cnn.add_module('dec_{}_{}'.format(i-1, i), l)
                else:
                    l = convlayer.TConv2D(in_ch, out_ch, kernel, stride, padding, output_padding, act=act,
                                          norm_layer=norm, adain_num_style_features=dim_z)  # TODO concat MIDI note ?
                    self.single_ch_cnn.add_module('dec{}'.format(i), l)
        # Final activation, for all architectures
        self.single_ch_cnn_act = nn.Hardtanh()

        # TODO send a dummy input to retrieve output size (assert if CNN output is smaller than target)

    def forward(self, z_sampled, w_style=None):  # TODO add midi_notes as arg
        if w_style is None:  # w_style can be omitted during summary writing
            print("[decoder.py] w_style tensor is None; replaced by 0.0 (should happen during init only)")
            w_style = z_sampled * 0.0
        mixed_features = self.mlp(z_sampled)
        mixed_features = mixed_features.view(-1,  # batch size auto inferred
                                             self.cnn_input_shape[0], self.cnn_input_shape[1], self.cnn_input_shape[2])
        unmixed_features, _ = self.features_unmixer_cnn((mixed_features, w_style))
        # This won't work on multi-channel spectrograms with force_bigger_network==True (don't do this anyway)
        single_ch_cnn_inputs = torch.split(unmixed_features, self.last_4x4conv_ch,  # actual multi-spec: 512ch-split
                                           dim=1)  # Split along channels dimension
        # TODO concat midi notes to w_style
        single_ch_cnn_outputs = list()
        for single_ch_in in single_ch_cnn_inputs:
            single_ch_cnn_outputs.append(self.single_ch_cnn((single_ch_in, w_style)))
        # Remove style and activate outputs
        single_ch_cnn_outputs = [self.single_ch_cnn_act(x[0]) for x in single_ch_cnn_outputs]

        return torch.cat(single_ch_cnn_outputs, dim=1)  # Concatenate all single-channel spectrograms


class SpectrogramCNN(nn.Module):
    """ A decoder CNN network for spectrogram output """

    def __init__(self, architecture, output_tensor_size: Tuple[int, int, int, int], output_activation=nn.Hardtanh(),
                 append_1x1_conv=True, force_bigger_network=False):
        """ Defines a decoder given the specified architecture. """
        super().__init__()
        self.architecture = architecture
        if not append_1x1_conv:   # Only these archs are fully-supported at the moment
            assert self.architecture == 'speccnn8l1_bn' or self.architecture == 'speccnn9l1' \
                   or self.architecture == 'rescnn'
        self.output_tensor_size = output_tensor_size
        assert self.output_tensor_size[1] == 1  # This decoder is single-channel output

        if self.architecture == 'wavenet_baseline':  # https://arxiv.org/abs/1704.01279
            ''' Symmetric layer output sizes (compared to the encoder).
            No activation and batch norm after the last up-conv.
            
            Issue: this architecture induces a huge number of ops within the 2 last layers.
            Unusable with reducing the spectrogram or getting 8 or 16 GPUs. '''
            self.dec_nn = nn.Sequential(convlayer.TConv2D(1024, 512, [1 , 1], [1 , 1], 0,
                                                          activation=nn.LeakyReLU(0.1), name_prefix='dec1'),
                                        convlayer.TConv2D(512, 512, [4, 4], [2, 1], 2, output_padding=[1, 0],
                                                          activation=nn.LeakyReLU(0.1), name_prefix='dec2'),
                                        convlayer.TConv2D(512, 256, [4, 4], [2, 2], 2, output_padding=[1, 1],
                                                          activation=nn.LeakyReLU(0.1), name_prefix='dec3'),
                                        convlayer.TConv2D(256, 256, [4, 4], [2, 2], 2, output_padding=[1, 0],
                                                          activation=nn.LeakyReLU(0.1), name_prefix='dec4'),
                                        convlayer.TConv2D(256, 256, [4, 4], [2, 2], 2, output_padding=[1, 1],
                                                          activation=nn.LeakyReLU(0.1), name_prefix='dec5'),
                                        convlayer.TConv2D(256, 128, [4, 4], [2, 2], 2, output_padding=[1, 0],
                                                          activation=nn.LeakyReLU(0.1), name_prefix='dec6'),
                                        convlayer.TConv2D(128, 128, [4, 4], [2, 2], 2, output_padding=[1, 1],
                                                          activation=nn.LeakyReLU(0.1), name_prefix='dec7'),
                                        convlayer.TConv2D(128, 128, [4, 4], [2, 2], 2, output_padding=[1, 1],
                                                          activation=nn.LeakyReLU(0.1), name_prefix='dec8'),
                                        convlayer.TConv2D(128, 128, [5, 5], [2, 2], 2, output_padding=[0, 0],
                                                          activation=nn.LeakyReLU(0.1), name_prefix='dec9'),
                                        nn.ConvTranspose2d(128, 1, [5, 5], [2, 2], 2)  # TODO bounded activation
                                        )

        elif self.architecture == 'wavenet_baseline_lighter':
            ''' Lighter decoder compared to wavenet baseline, but keeps an acceptable number
            of GOPs for last transpose-conv layers '''
            self.dec_nn = nn.Sequential(convlayer.TConv2D(1024, 512, [1 , 1], [1 , 1], 0,
                                                          activation=nn.LeakyReLU(0.1), name_prefix='dec1'),
                                        convlayer.TConv2D(512, 512, [4, 4], [2, 1], 2, output_padding=[1, 0],
                                                          activation=nn.LeakyReLU(0.1), name_prefix='dec2'),
                                        convlayer.TConv2D(512, 256, [4, 4], [2, 2], 2, output_padding=[1, 1],
                                                          activation=nn.LeakyReLU(0.1), name_prefix='dec3'),
                                        convlayer.TConv2D(256, 256, [4, 4], [2, 2], 2, output_padding=[1, 0],
                                                          activation=nn.LeakyReLU(0.1), name_prefix='dec4'),
                                        convlayer.TConv2D(256, 256, [4, 4], [2, 2], 2, output_padding=[1, 1],
                                                          activation=nn.LeakyReLU(0.1), name_prefix='dec5'),
                                        convlayer.TConv2D(256, 128, [4, 4], [2, 2], 2, output_padding=[1, 0],
                                                          activation=nn.LeakyReLU(0.1), name_prefix='dec6'),
                                        convlayer.TConv2D(128, 64, [4, 4], [2, 2], 2, output_padding=[1, 1],
                                                          activation=nn.LeakyReLU(0.1), name_prefix='dec7'),
                                        convlayer.TConv2D(64, 32, [4, 4], [2, 2], 2, output_padding=[1, 1],
                                                          activation=nn.LeakyReLU(0.1), name_prefix='dec8'),
                                        convlayer.TConv2D(32, 16, [5, 5], [2, 2], 2, output_padding=[0, 0],
                                                          activation=nn.LeakyReLU(0.1), name_prefix='dec9'),
                                        nn.ConvTranspose2d(16, 1, [5, 5], [2, 2], 2)  # TODO bounded activation
                                        )

        elif self.architecture == 'wavenet_baseline_shallow':  # Inspired from wavenet_baseline
            self.dec_nn = nn.Sequential(convlayer.TConv2D(1024, 512, [1 , 1], [1 , 1], 0,
                                                          activation=nn.LeakyReLU(0.1), name_prefix='dec1'),
                                        convlayer.TConv2D(512, 256, [4, 4], [2, 2], 2, output_padding=[1, 0],
                                                          activation=nn.LeakyReLU(0.1), name_prefix='dec2'),
                                        convlayer.TConv2D(256, 128, [4, 4], [2, 2], 2, output_padding=[1, 1],
                                                          activation=nn.LeakyReLU(0.1), name_prefix='dec3'),
                                        convlayer.TConv2D(128, 64, [4, 4], [2, 2], 2, output_padding=[1, 0],
                                                          activation=nn.LeakyReLU(0.1), name_prefix='dec4'),
                                        convlayer.TConv2D(64, 32, [4, 4], [2, 2], 2, output_padding=[1, 1],
                                                          activation=nn.LeakyReLU(0.1), name_prefix='dec5'),
                                        convlayer.TConv2D(32, 16, [4, 4], [2, 2], 2, output_padding=[1, 1],
                                                          activation=nn.LeakyReLU(0.1), name_prefix='dec6'),
                                        convlayer.TConv2D(16, 8, [4, 4], [2, 2], 2, output_padding=[1, 1],
                                                          activation=nn.LeakyReLU(0.1), name_prefix='dec7'),
                                        nn.ConvTranspose2d(8, 1, [5, 5], [2, 2], 2)  # TODO bounded activation
                                        )

        elif self.architecture == 'flow_synth':
            ''' This decoder seems as GPU-heavy as wavenet_baseline?? '''
            n_lay = 64  # 128/2 for paper's comparisons consistency. Could be larger
            k7 = [7, 7]  # Kernel of size 7
            if self.output_tensor_size == (513, 433):
                pads = [3, 3, 3, 3, 2]  # FIXME
                out_pads = None
            elif self.output_tensor_size == (257, 347):  # 7.7 GB (RAM), 6.0 GMultAdd (batch 256) (inc. linear layers)
                pads = [3, 3, 3, 3, 2]
                out_pads = [0, [1, 0], [0, 1], [1, 0]]  # No output padding on last layer
            self.dec_nn = nn.Sequential(convlayer.TConv2D(n_lay, n_lay, k7, [2, 2], pads[0], out_pads[0], [2, 2],
                                                          activation=nn.ELU(), name_prefix='dec1'),
                                        convlayer.TConv2D(n_lay, n_lay, k7, [2, 2], pads[1], out_pads[1], [2, 2],
                                                          activation=nn.ELU(), name_prefix='dec2'),
                                        convlayer.TConv2D(n_lay, n_lay, k7, [2, 2], pads[2], out_pads[2], [2, 2],
                                                          activation=nn.ELU(), name_prefix='dec3'),
                                        convlayer.TConv2D(n_lay, n_lay, k7, [2, 2], pads[3], out_pads[3], [2, 2],
                                                          activation=nn.ELU(), name_prefix='dec4'),
                                        nn.ConvTranspose2d(n_lay, 1, k7, [2, 2], pads[4]),
                                        output_activation
                                        )

        elif self.architecture == 'speccnn8l1'\
            or self.architecture == 'speccnn8l1_bn':  # 1.8 GB (RAM) ; 0.36 GMultAdd  (batch 256)
            ''' Inspired by the wavenet baseline spectral autoencoder, but all sizes are drastically reduced
            (especially the last, not so useful but very GPU-expensive layers on large images) '''
            act = nn.LeakyReLU
            act_p = 0.1  # Activation param
            self.dec_nn = nn.Sequential(convlayer.TConv2D((512 if not force_bigger_network else 1800),
                                                          256, [4, 4], [2, 2], 2, output_padding=[1, 1],
                                                          activation=act(act_p), name_prefix='dec2'),
                                        convlayer.TConv2D(256, 128, [4, 4], [2, 2], 2, output_padding=[1, 0],
                                                          activation=act(act_p), name_prefix='dec3'),
                                        convlayer.TConv2D(128, 64, [4, 4], [2, 2], 2, output_padding=[1, 1],
                                                          activation=act(act_p), name_prefix='dec4'),
                                        convlayer.TConv2D(64, 32, [4, 4], [2, 2], 2, output_padding=[1, 1],
                                                          activation=act(act_p), name_prefix='dec5'),
                                        convlayer.TConv2D(32, 16, [4, 4], [2, 2], 2, output_padding=[1, 0],
                                                          activation=act(act_p), name_prefix='dec6'),
                                        convlayer.TConv2D(16, 8, [4, 4], [2, 2], 2, output_padding=[1, 0],
                                                          activation=act(act_p), name_prefix='dec7'),
                                        nn.ConvTranspose2d(8, 1, [5, 5], [2, 2], 2),
                                        output_activation
                                        )
            if append_1x1_conv:  # 1x1 "un-mixing" conv inserted as first conv layer
                assert False  # FIXME 1024ch should not be constant
                self.dec_nn = nn.Sequential(convlayer.TConv2D(1024, 512, [1 , 1], [1 , 1], 0,
                                                              activation=act(act_p), name_prefix='dec1'),
                                            self.dec_nn)

        elif self.architecture == 'speccnn8l1_2':  # 5.8 GB (RAM) ; 2.4 GMultAdd  (batch 256)
            act = nn.LeakyReLU
            act_p = 0.1  # Activation param
            self.dec_nn = nn.Sequential(convlayer.TConv2D(1024, 512, [1, 1], [1, 1], 0,
                                                          activation=act(act_p), name_prefix='dec1'),
                                        convlayer.TConv2D(512, 256, [4, 4], [2, 2], 2, output_padding=[1, 1],
                                                          activation=act(act_p), name_prefix='dec2'),
                                        convlayer.TConv2D(256, 256, [4, 4], [2, 2], 2, output_padding=[1, 0],
                                                          activation=act(act_p), name_prefix='dec3'),
                                        convlayer.TConv2D(256, 128, [4, 4], [2, 2], 2, output_padding=[1, 1],
                                                          activation=act(act_p), name_prefix='dec4'),
                                        convlayer.TConv2D(128, 128, [4, 4], [2, 2], 2, output_padding=[1, 1],
                                                          activation=act(act_p), name_prefix='dec5'),
                                        convlayer.TConv2D(128, 64, [4, 4], [2, 2], 2, output_padding=[1, 0],
                                                          activation=act(act_p), name_prefix='dec6'),
                                        convlayer.TConv2D(64, 32, [4, 4], [2, 2], 2, output_padding=[1, 0],
                                                          activation=act(act_p), name_prefix='dec7'),
                                        nn.ConvTranspose2d(32, 1, [5, 5], [2, 2], 2),
                                        output_activation
                                        )
        elif self.architecture == 'speccnn8l1_3':  # XXX GB (RAM) ; XXX GMultAdd  (batch 256)
            ''' Inspired by the wavenet baseline spectral autoencoder, but all sizes drastically reduced '''
            act = nn.LeakyReLU
            act_p = 0.1  # Activation param
            ker = [5, 5]
            self.dec_nn = nn.Sequential(convlayer.TConv2D(1024, 512, [1, 1], [1, 1], 0,
                                                          activation=act(act_p), name_prefix='dec1'),
                                        convlayer.TConv2D(512, 256, ker, [2, 2], 2, output_padding=[0, 1],
                                                          activation=act(act_p), name_prefix='dec2'),
                                        convlayer.TConv2D(256, 128, ker, [2, 2], 2, output_padding=[0, 0],
                                                          activation=act(act_p), name_prefix='dec3'),
                                        convlayer.TConv2D(128, 64, ker, [2, 2], 2, output_padding=[0, 1],
                                                          activation=act(act_p), name_prefix='dec4'),
                                        convlayer.TConv2D(64, 32, ker, [2, 2], 2, output_padding=[0, 1],
                                                          activation=act(act_p), name_prefix='dec5'),
                                        convlayer.TConv2D(32, 16, ker, [2, 2], 2, output_padding=[0, 0],
                                                          activation=act(act_p), name_prefix='dec6'),
                                        convlayer.TConv2D(16, 8, ker, [2, 2], 2, output_padding=[0, 1],
                                                          activation=act(act_p), name_prefix='dec7'),
                                        nn.ConvTranspose2d(8, 1, [5, 5], [2, 2], 2),
                                        output_activation
                                        )

        elif self.architecture == 'speccnn9l1':  # 9.1 GB (RAM) ; 2.9 GMultAdd  (batch 256)
            # TODO description and implementation
            act = nn.LeakyReLU
            act_p = 0.1  # Activation param
            self.dec_nn = nn.Sequential(convlayer.TConv2D((512 if not force_bigger_network else 1800),
                                                          256, [4, 4], [2, 2], 2, output_padding=[1, 0],
                                                          activation=act(act_p), name_prefix='dec2'),
                                        convlayer.TConv2D(256, 256, [4, 4], [2, 2], 2, output_padding=[1, 1],
                                                          activation=act(act_p), name_prefix='dec3'),
                                        convlayer.TConv2D(256, 128, [4, 4], [2, 2], 2, output_padding=[1, 1],
                                                          activation=act(act_p), name_prefix='dec4'),
                                        convlayer.TConv2D(128, 128, [4, 4], [2, 2], 2, output_padding=[1, 0],
                                                          activation=act(act_p), name_prefix='dec5'),
                                        convlayer.TConv2D(128, 96, [4, 4], [2, 2], 2, output_padding=[1, 0],
                                                          activation=act(act_p), name_prefix='dec6'),
                                        convlayer.TConv2D(96, 48, [4, 4], [2, 2], 2, output_padding=[1, 0],
                                                          activation=act(act_p), name_prefix='dec7'),
                                        convlayer.TConv2D(48, 24, [4, 4], [2, 2], 2, output_padding=[1, 0],
                                                          activation=act(act_p), name_prefix='dec8'),
                                        nn.ConvTranspose2d(24, 1, (5, 5), (2, 2), (2, 1)),
                                        output_activation
                                        )
            if append_1x1_conv:  # 1x1 "un-mixing" conv inserted as first conv layer
                assert False  # FIXME 1024ch should not be constant
                self.dec_nn = nn.Sequential(convlayer.TConv2D(1024, 512, [1 , 1], [1 , 1], 0,
                                                              activation=act(act_p), name_prefix='dec1'),
                                            self.dec_nn)

        elif self.architecture == 'rescnn':  # XXX GB (RAM) ; XXX GMultAdd  (batch 256)
            # TODO description and implementation
            act = nn.LeakyReLU
            act_p = 0.1  # Activation param
            self.dec_nn = nn.Sequential(convlayer.TConv2D((512 if not force_bigger_network else 1800),
                                                          256, [4, 4], [2, 2], 2, output_padding=[1, 0],
                                                          activation=act(act_p), name_prefix='dec2'),
                                        convlayer.ResTConv2D(256, 256, [4, 4], [2, 2], 2, output_padding=[1, 1],
                                                             activation=act(act_p), name_prefix='dec34'),
                                        convlayer.DenseTConv2D(256, 128, 32, 96, [4, 4], [2, 2], 2, output_padding=[1, 0],
                                                               activation=act(act_p), name_prefix='dec56'),
                                        convlayer.DenseTConv2D(96, 48, 8, 32, [4, 4], [2, 2], 2, output_padding=[1, 0],
                                                               activation=act(act_p), name_prefix='dec78'),
                                        nn.ConvTranspose2d(32, 1, (5, 5), (2, 2), (2, 1)),
                                        output_activation
                                        )
            if append_1x1_conv:  # 1x1 "un-mixing" conv inserted as first conv layer
                assert False  # FIXME 1024ch should not be constant
                self.dec_nn = nn.Sequential(convlayer.TConv2D(1024, 512, [1 , 1], [1 , 1], 0,
                                                              activation=act(act_p), name_prefix='dec1'),
                                            self.dec_nn)

        else:
            raise NotImplementedError("Architecture '{}' not available".format(self.architecture))

    def forward(self, x_spectrogram):
        x_hat_raw = self.dec_nn(x_spectrogram)
        # Crop if necessary - TODO disable this during development (to know the actual convolutions output size)
        # TODO does this actually prevent autograd tracer warnings?

        # TODO manually crop (depending on spectrogram size) to prevent autograd warnings?

        x_hat_H, x_hat_W = int(x_hat_raw.shape[2]), int(x_hat_raw.shape[3])  # 1st dim: number of lines
        if (x_hat_H, x_hat_W) != (self.output_tensor_size[2], self.output_tensor_size[3]):
            left_margin = (x_hat_W - self.output_tensor_size[3]) // 2
            top_margin = (x_hat_H - self.output_tensor_size[2]) // 2
            assert top_margin >= 0 and left_margin >= 0
            return x_hat_raw[:, :, top_margin:top_margin+self.output_tensor_size[2],
                             left_margin:left_margin+self.output_tensor_size[3]]
        else:
            return x_hat_raw


if __name__ == "__main__":

    import torchinfo
    output_size = (32, 1, 257, 347)
    dec = SpectrogramDecoder('speccnn8l1', 610, output_size, 0.1)
    _ = torchinfo.summary(dec, input_size=(32, 610))

