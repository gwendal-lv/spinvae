import warnings

import torch
import torch.nn as nn

from model import convlayer


def parse_architecture(full_architecture: str):
    """ Parses an argument used to describe the encoder and decoder architectures (e.g. speccnn8l1_big_res) """
    # Decompose architecture to retrieve number of cnn and fc layer, options, ...
    arch_args = full_architecture.split('_')
    base_arch_name = arch_args[0]  # type: str
    del arch_args[0]
    if base_arch_name.startswith('speccnn'):
        layers_args = [int(s) for s in base_arch_name.replace('speccnn', '').split('l')]
    else:
        raise AssertionError("Base architecture not available for given arch '{}'".format(base_arch_name))
    num_cnn_layers = layers_args[0]
    num_fc_layers = layers_args[1]
    # Check arch args, transform
    arch_args_dict = {'adain': False, 'big': False, 'res': False, 'att': False}
    for arch_arg in arch_args:
        if arch_arg in ['adain', 'big', 'res']:
            arch_args_dict[arch_arg] = True  # Authorized arguments
        elif arch_arg == 'att':
            raise NotImplementedError("Self-attention encoder argument (_att) not implemented.")
        else:
            raise ValueError("Unvalid encoder argument '{}' from architecture '{}'".format(arch_arg, full_architecture))
    return base_arch_name, num_cnn_layers, num_fc_layers, arch_args_dict


class SpectrogramEncoder(nn.Module):
    """ Contains a spectrogram-input CNN and some MLP layers, and outputs the mu and logs(var) values"""
    def __init__(self, architecture, dim_z, input_tensor_size, fc_dropout, output_bn=False, output_dropout_p=0.0,
                 deep_features_mix_level=-1, force_bigger_network=False):
        """

        :param architecture: String describing the type of network, number of CNN and FC layers, with options
            (see parse_architecture(...) method)
        :param dim_z:
        :param input_tensor_size:
        :param fc_dropout:
        :param output_bn:
        :param deep_features_mix_level: (applies to multi-channel spectrograms only) negative int, -1 corresponds
            to the last conv layer (usually 1x1 kernel), -2 to the conv layer before, etc.
        :param force_bigger_network: Optional, to impose a higher number of channels for the last 4x4 (should be
            used only for fair comparisons between single/multi-specs encoder)
        """
        super().__init__()
        self.dim_z = dim_z  # Latent-vector size (2*dim_z encoded values - mu and logs sigma 2)
        self.spectrogram_channels = input_tensor_size[1]
        self.full_architecture = architecture
        self.deep_feat_mix_level = deep_features_mix_level
        self.fc_dropout = fc_dropout
        if force_bigger_network :  # deactivated after refactoring
            raise NotImplementedError()
        self.base_arch_name, self.num_cnn_layers, self.num_fc_layers, self.arch_args \
            = parse_architecture(self.full_architecture)

        # - - - - - 1) Main CNN encoder (applied once per input spectrogram channel) - - - - -
        # - - - - - - - - - - and 2) Features mixer - - - - - - - - - -
        self.single_ch_cnn = nn.Sequential()
        self.features_mixer_cnn = nn.Sequential()
        self._build_cnns()

        # - - - - - 3) MLP for extracting properly-sized latent vector - - - - -
        # Automatic CNN output tensor size inference
        with torch.no_grad():
            single_element_input_tensor_size = list(input_tensor_size)
            single_element_input_tensor_size[0] = 1  # single-element batch
            dummy_spectrogram = torch.zeros(single_element_input_tensor_size)
            self.cnn_out_size = self._forward_cnns(dummy_spectrogram).size()
        cnn_out_items = self.cnn_out_size[1] * self.cnn_out_size[2] * self.cnn_out_size[3]
        # Number of linear layers as configured in the arch arg (e.g. speccnn8l1 -> 1 FC layer).
        # Default: no final batch-norm (maybe added after this for loop). Always 1024 hidden units
        self.mlp = nn.Sequential()
        for i in range(self.num_fc_layers):
            if self.fc_dropout > 0.0:
                self.mlp.add_module("encdrop{}".format(i), nn.Dropout(self.fc_dropout))
            in_units = cnn_out_items if (i == 0) else 1024
            out_units = 1024 if (i < (self.num_fc_layers - 1)) else 2 * self.dim_z
            self.mlp.add_module("encfc{}".format(i), nn.Linear(in_units, out_units))
            if i < (self.num_fc_layers - 1):  # No final activation - outputs are latent mu/logvar
                self.mlp.add_module("act{}".format(i), nn.ReLU())
        # Batch-norm here to compensate for unregularized z0 of a flow-based latent space (replace 0.1 Dkl)
        # Dropout to help prevent VAE posterior collapse --> should be zero
        if output_bn:
            self.mlp.add_module('lat_in_regularization', nn.BatchNorm1d(2 * self.dim_z))
        if output_dropout_p > 0.0:
            self.mlp.add_module('lat_in_drop', nn.Dropout(output_dropout_p))

    def _build_cnns(self):
        if self.base_arch_name == 'speccnn8l1':
            ''' Where to use BN? 'ESRGAN' generator does not use BN in the first and last conv layers.
            DCGAN: no BN on discriminator in out generator out.
            Our experiments seem to show: more stable latent loss with no BN before the FC that regresses mu/logvar,
            consistent training runs  '''
            if self.arch_args['adain']:
                print("[encoder.py] 'adain' arch arg (MIDI notes provided to layers) not implemented")
            _in_ch, _out_ch = -1, -1  # backup for res blocks
            building_res_block = False  # if True, the current layer will be added from the next iteration
            finish_res_block = False  # if True, the current layer will include the previous one (res block)
            for i in range(0, self.num_cnn_layers):
                if self.arch_args['res']:
                    if i == 1 or i == 3 or (i == 5 and self.deep_feat_mix_level > -2):
                        building_res_block, finish_res_block = True, False
                    elif i == 2 or i == 4 or (i == 6 and self.deep_feat_mix_level > -2):
                        building_res_block, finish_res_block = False, True
                    else:
                        building_res_block, finish_res_block = False, False
                # num ch, kernel size, stride and padding depend on the layer number
                # base number of channels: 1, 8, 16, ... 512, 1024
                if i == 0:
                    kernel_size, stride, padding = [5, 5], [2, 2], 2
                    in_ch = 1
                    out_ch = 2**(i+3)
                    if self.arch_args['big']:
                        out_ch = max(out_ch, 128)
                elif 1 <= i <= 6:
                    kernel_size, stride, padding = [4, 4], [2, 2], 2
                    in_ch = 2**(i+2)
                    out_ch = 2**(i+3)
                    if self.arch_args['big']:
                        in_ch, out_ch = max(in_ch, 128), max(out_ch, 128)
                else:  # i == 7
                    kernel_size, stride, padding = [1, 1], [1, 1], 0
                    in_ch = 2**(i+2)
                    out_ch = 2**(i+3)
                # Increased number of layers - sequential encoder
                if self.spectrogram_channels > 1:
                    # Does this layer receive the stacked feature maps? (much larger input, 50% larger output)
                    if i == (self.num_cnn_layers + self.deep_feat_mix_level):  # negative mix level
                        in_ch = in_ch * self.spectrogram_channels
                        out_ch = out_ch * 3 // 2
                    # Is this layer the one that is after the stacking layer? (50% larger input)
                    elif i == (self.num_cnn_layers + self.deep_feat_mix_level + 1):  # negative mix level
                        in_ch = in_ch * 3 // 2
                # Build layer and append to the appropriate sequence module
                act = nn.LeakyReLU(0.1)  # New instance for each layer (maybe unnecessary?)
                # No normalization on first and last layers
                if 0 < i < (self.num_cnn_layers - 1):
                    # TODO activate AdaIn?
                    norm = 'bn'
                else:
                    norm = None
                if building_res_block:
                    _in_ch, _out_ch = in_ch, out_ch
                else:
                    if finish_res_block:
                        name = 'enc_{}_{}'.format(i-1, i)
                        conv_layer = convlayer.ResConv2D(_in_ch, in_ch, out_ch, kernel_size, stride, padding,
                                                         act=act, norm_layer=norm, adain_num_style_features=None)
                    else:
                        name = 'enc{}'.format(i)
                        conv_layer = convlayer.Conv2D(in_ch, out_ch, kernel_size, stride, padding,
                                                      act=act, norm_layer=norm, adain_num_style_features=None)
                    if i < (self.num_cnn_layers + self.deep_feat_mix_level):  # negative mix level
                        self.single_ch_cnn.add_module(name, conv_layer)
                    else:
                        self.features_mixer_cnn.add_module(name, conv_layer)
        else:
            raise AssertionError("Architecture {} not available".format(self.base_arch_name))

    def _forward_cnns(self, x_spectrograms):
        # TODO use MIDI notes (through AdaIN style?)
        # apply main cnn multiple times
        single_channel_cnn_out = [self.single_ch_cnn((torch.unsqueeze(x_spectrograms[:, ch, :, :], dim=1), None))
                                  for ch in range(self.spectrogram_channels)]
        # Remove w output (sequential module: conditioning passed to all layers)
        single_channel_cnn_out = [x[0] for x in single_channel_cnn_out]
        # Then mix features from different input channels
        x_out, w = self.features_mixer_cnn((torch.cat(single_channel_cnn_out, dim=1), None))
        return x_out

    def forward(self, x_spectrograms):
        n_minibatch = x_spectrograms.size()[0]
        cnn_out = self._forward_cnns(x_spectrograms).view(n_minibatch, -1)  # 2nd dim automatically inferred
        # print("Forward CNN out size = {}".format(cnn_out.size()))
        z_mu_logvar = self.mlp(cnn_out)
        # Last dim contains a latent proba distribution value, last-1 dim is 2 (to retrieve mu or logs sigma2)
        return torch.reshape(z_mu_logvar, (n_minibatch, 2, self.dim_z))


class SpectrogramCNN(nn.Module):
    """ A encoder CNN network for spectrogram input """

    # TODO Option to enable res skip connections
    # TODO Option to choose activation function
    def __init__(self, architecture, last_layers_to_remove=0):
        """
        Automatically defines an autoencoder given the specified architecture

        :param last_layers_to_remove: Number of deepest conv layers to omit in this module (they will be added in
            the owner of this pure-CNN module).
        """
        super().__init__()
        assert False  # Obsolete class - code to be removed
        self.architecture = architecture
        if last_layers_to_remove > 0:  # Only these archs are fully-supported at the moment
            assert self.architecture == 'speccnn8l1_bn' or self.architecture == 'speccnn9l1' \
                   or self.architecture == 'rescnn'

        if self.architecture == 'wavenet_baseline'\
           or self.architecture == 'wavenet_baseline_lighter':  # this encoder is quite light already
            # TODO adapt to smaller spectrograms
            ''' Based on strided convolutions - no max pool (reduces the total amount of
             conv operations).  https://arxiv.org/abs/1704.01279
             No dilation: the receptive field in enlarged through a larger number
             of layers. 
             Layer 8 has a lower time-stride (better time resolution).
             Size of layer 9 (1024 ch) corresponds the wavenet time-encoder.
             
             Issue: when using the paper's FFT size and hop, layers 8 and 9 seem less useful. The image size
              at this depth is < kernel size (much of the 4x4 kernel convolves with zeros) '''
            self.enc_nn = nn.Sequential(convlayer.Conv2D(1, 128, [5, 5], [2, 2], 2, [1, 1],
                                                         act=nn.LeakyReLU(0.1), name_prefix='enc1'),
                                        convlayer.Conv2D(128, 128, [4, 4], [2, 2], 2, [1, 1],
                                                         act=nn.LeakyReLU(0.1), name_prefix='enc2'),
                                        convlayer.Conv2D(128, 128, [4, 4], [2, 2], 2, [1, 1],
                                                         act=nn.LeakyReLU(0.1), name_prefix='enc3'),
                                        convlayer.Conv2D(128, 256, [4, 4], [2, 2], 2, [1, 1],
                                                         act=nn.LeakyReLU(0.1), name_prefix='enc4'),
                                        convlayer.Conv2D(256, 256, [4, 4], [2, 2], 2, [1, 1],
                                                         act=nn.LeakyReLU(0.1), name_prefix='enc5'),
                                        convlayer.Conv2D(256, 256, [4, 4], [2, 2], 2, [1, 1],
                                                         act=nn.LeakyReLU(0.1), name_prefix='enc6'),
                                        convlayer.Conv2D(256, 512, [4, 4], [2, 2], 2, [1, 1],
                                                         act=nn.LeakyReLU(0.1), name_prefix='enc7'),
                                        convlayer.Conv2D(512, 512, [4, 4], [2, 2], 2, [1, 1],
                                                         act=nn.LeakyReLU(0.1), name_prefix='enc8'),
                                        convlayer.Conv2D(512, 512, [4, 4], [2, 1], 2, [1, 1],
                                                         act=nn.LeakyReLU(0.1), name_prefix='enc9'),
                                        convlayer.Conv2D(512, 1024, [1, 1], [1, 1], 0, [1, 1],
                                                         act=nn.LeakyReLU(0.1), name_prefix='enc10'),
                                        )

        elif self.architecture == 'wavenet_baseline_shallow':
            """ Inspired from wavenet_baseline, minus the two last layer, with less channels """
            self.enc_nn = nn.Sequential(convlayer.Conv2D(1, 8, [5, 5], [2, 2], 2, [1, 1],
                                                         act=nn.LeakyReLU(0.1), name_prefix='enc1'),
                                        convlayer.Conv2D(8, 16, [4, 4], [2, 2], 2, [1, 1],
                                                         act=nn.LeakyReLU(0.1), name_prefix='enc2'),
                                        convlayer.Conv2D(16, 32, [4, 4], [2, 2], 2, [1, 1],
                                                         act=nn.LeakyReLU(0.1), name_prefix='enc3'),
                                        convlayer.Conv2D(32, 64, [4, 4], [2, 2], 2, [1, 1],
                                                         act=nn.LeakyReLU(0.1), name_prefix='enc4'),
                                        convlayer.Conv2D(64, 128, [4, 4], [2, 2], 2, [1, 1],
                                                         act=nn.LeakyReLU(0.1), name_prefix='enc5'),
                                        convlayer.Conv2D(128, 256, [4, 4], [2, 2], 2, [1, 1],
                                                         act=nn.LeakyReLU(0.1), name_prefix='enc6'),
                                        convlayer.Conv2D(256, 512, [4, 4], [2, 2], 2, [1, 1],
                                                         act=nn.LeakyReLU(0.1), name_prefix='enc7'),
                                        convlayer.Conv2D(512, 1024, [1, 1], [1, 1], 0, [1, 1],
                                                         act=nn.LeakyReLU(0.1), name_prefix='enc8'),
                                        )

        elif self.architecture == 'flow_synth':
            # spectrogram (257, 347):   7.7 GB (RAM), 1.4 GMultAdd (batch 256) (inc. linear layers)
            ''' https://acids-ircam.github.io/flow_synthesizer/#models-details
            Based on strided convolutions and dilation to quickly enlarge the receptive field.
            Paper says: "5 layers with 128 channels of strided dilated 2-D convolutions with kernel
            size 7, stride 2 and an exponential dilation factor of 2l (starting at l=0) with batch
            normalization and ELU activation." Code from their git repo:
            dil = ((args.dilation == 3) and (2 ** l) or args.dilation)
            pad = 3 * (dil + 1)
            
            Potential issue: this dilation is extremely big for deep layers 4 and 5. Dilated kernel is applied
            mostly on zero-padded values. We should either stride-conv or 2^l dilate, but not both '''
            n_lay = 64  # 128/2 for paper's comparisons consistency. Could be larger
            self.enc_nn = nn.Sequential(convlayer.Conv2D(1, n_lay, [7, 7], [2, 2], 3, [1, 1],
                                                         act=nn.ELU(), name_prefix='enc1'),
                                        convlayer.Conv2D(n_lay, n_lay, [7, 7], [2, 2], 3, [2, 2],
                                                         act=nn.ELU(), name_prefix='enc2'),
                                        convlayer.Conv2D(n_lay, n_lay, [7, 7], [2, 2], 3, [2, 2],
                                                         act=nn.ELU(), name_prefix='enc3'),
                                        convlayer.Conv2D(n_lay, n_lay, [7, 7], [2, 2], 3, [2, 2],
                                                         act=nn.ELU(), name_prefix='enc4'),
                                        convlayer.Conv2D(n_lay, n_lay, [7, 7], [2, 2], 3, [2, 2],
                                                         act=nn.ELU(), name_prefix='enc5'))

        elif self.architecture == 'speccnn8l1':  # 1.7 GB (RAM) ; 0.12 GMultAdd  (batch 256)
            ''' Inspired by the wavenet baseline spectral autoencoder, but all sizes drastically reduced.
            Where to use BN? 'Super-Resolution GAN' generator does not use BN in the first and last conv layers.'''
            act = nn.LeakyReLU
            act_p = 0.1  # Activation param
            self.enc_nn = nn.Sequential(convlayer.Conv2D(1, 8, [5, 5], [2, 2], 2, [1, 1],
                                                         act=act(act_p), name_prefix='enc1'),
                                        convlayer.Conv2D(8, 16, [4, 4], [2, 2], 2, [1, 1],
                                                         act=act(act_p), name_prefix='enc2'),
                                        convlayer.Conv2D(16, 32, [4, 4], [2, 2], 2, [1, 1],
                                                         act=act(act_p), name_prefix='enc3'),
                                        convlayer.Conv2D(32, 64, [4, 4], [2, 2], 2, [1, 1],
                                                         act=act(act_p), name_prefix='enc4'),
                                        convlayer.Conv2D(64, 128, [4, 4], [2, 2], 2, [1, 1],
                                                         act=act(act_p), name_prefix='enc5'),
                                        convlayer.Conv2D(128, 256, [4, 4], [2, 2], 2, [1, 1],
                                                         act=act(act_p), name_prefix='enc6'),
                                        convlayer.Conv2D(256, 512, [4, 4], [2, 2], 2, [1, 1],
                                                         act=act(act_p), name_prefix='enc7'),
                                        convlayer.Conv2D(512, 1024, [1, 1], [1, 1], 0, [1, 1],
                                                         act=act(act_p), name_prefix='enc8'),
                                        )


        # TODO res-blocks add (avg-pool?)
        # TODO stride 2 OR dilation 2, not both (a lot of spectrogram pixels are unused...)
        elif self.architecture == 'speccnn8l1_bn':  # 1.7 GB (RAM) ; 0.12 GMultAdd  (batch 256)
            ''' Where to use BN? 'ESRGAN' generator does not use BN in the first and last conv layers.
            DCGAN: no BN on discriminator in out generator out.
            Our experiments seem to show: more stable latent loss with no BN before the FC that regresses mu/logvar,
            consistent training runs 
            TODO try BN before act (see DCGAN arch)
            '''
            act = nn.LeakyReLU
            act_p = 0.1  # Activation param
            self.enc_nn = nn.Sequential(convlayer.Conv2D(1, 8, [5, 5], [2, 2], 2, [1, 1], batch_norm=None,
                                                         act=act(act_p), name_prefix='enc1'),
                                        convlayer.Conv2D(8, 16, [4, 4], [2, 2], 2, [1, 1],
                                                         act=act(act_p), name_prefix='enc2'),
                                        convlayer.Conv2D(16, 32, [4, 4], [2, 2], 2, [1, 1],
                                                         act=act(act_p), name_prefix='enc3'),
                                        convlayer.Conv2D(32, 64, [4, 4], [2, 2], 2, [1, 1],
                                                         act=act(act_p), name_prefix='enc4'),
                                        convlayer.Conv2D(64, 128, [4, 4], [2, 2], 2, [1, 1],
                                                         act=act(act_p), name_prefix='enc5'),
                                        convlayer.Conv2D(128, 256, [4, 4], [2, 2], 2, [1, 1],
                                                         act=act(act_p), name_prefix='enc6')
                                        )
            if last_layers_to_remove <= 1:
                self.enc_nn.add_module('4x4conv', convlayer.Conv2D(256, 512, [4, 4], [2, 2], 2, [1, 1],
                                                                   act=act(act_p), name_prefix='enc7'))
            if last_layers_to_remove == 0:
                self.enc_nn.add_module('1x1conv', convlayer.Conv2D(512, 1024, [1, 1], [1, 1], 0, [1, 1], batch_norm=None,
                                                                   act=act(act_p), name_prefix='enc8'))
        elif self.architecture == 'speccnn8l1_2':  # 5.8 GB (RAM) ; 0.65 GMultAdd  (batch 256)
            act = nn.LeakyReLU
            act_p = 0.1  # Activation param
            self.enc_nn = nn.Sequential(convlayer.Conv2D(1, 32, [5, 5], [2, 2], 2, [1, 1], batch_norm=None,
                                                         act=act(act_p), name_prefix='enc1'),
                                        convlayer.Conv2D(32, 64, [4, 4], [2, 2], 2, [1, 1],
                                                         act=act(act_p), name_prefix='enc2'),
                                        convlayer.Conv2D(64, 128, [4, 4], [2, 2], 2, [1, 1],
                                                         act=act(act_p), name_prefix='enc3'),
                                        convlayer.Conv2D(128, 128, [4, 4], [2, 2], 2, [1, 1],
                                                         act=act(act_p), name_prefix='enc4'),
                                        convlayer.Conv2D(128, 256, [4, 4], [2, 2], 2, [1, 1],
                                                         act=act(act_p), name_prefix='enc5'),
                                        convlayer.Conv2D(256, 256, [4, 4], [2, 2], 2, [1, 1],
                                                         act=act(act_p), name_prefix='enc6'),
                                        convlayer.Conv2D(256, 512, [4, 4], [2, 2], 2, [1, 1],
                                                         act=act(act_p), name_prefix='enc7'),
                                        convlayer.Conv2D(512, 1024, [1, 1], [1, 1], 0, [1, 1], batch_norm=None,
                                                         act=act(act_p), name_prefix='enc8'),
                                        )
        elif self.architecture == 'speccnn8l1_3':  # XXX GB (RAM) ; XXX GMultAdd  (batch 256)
            ''' speeccnn8l1_bn with bigger conv kernels '''
            act = nn.LeakyReLU
            act_p = 0.1  # Activation param
            ker = [5, 5]  # TODO try bigger 1st ker?
            self.enc_nn = nn.Sequential(convlayer.Conv2D(1, 8, [5, 5], [2, 2], 2, [1, 1], batch_norm=None,
                                                         act=act(act_p), name_prefix='enc1'),
                                        convlayer.Conv2D(8, 16, ker, [2, 2], 2, [1, 1],
                                                         act=act(act_p), name_prefix='enc2'),
                                        convlayer.Conv2D(16, 32, ker, [2, 2], 2, [1, 1],
                                                         act=act(act_p), name_prefix='enc3'),
                                        convlayer.Conv2D(32, 64, ker, [2, 2], 2, [1, 1],
                                                         act=act(act_p), name_prefix='enc4'),
                                        convlayer.Conv2D(64, 128, ker, [2, 2], 2, [1, 1],
                                                         act=act(act_p), name_prefix='enc5'),
                                        convlayer.Conv2D(128, 256, ker, [2, 2], 2, [1, 1],
                                                         act=act(act_p), name_prefix='enc6'),
                                        convlayer.Conv2D(256, 512, ker, [2, 2], 2, [1, 1],
                                                         act=act(act_p), name_prefix='enc7'),
                                        convlayer.Conv2D(512, 1024, [1, 1], [1, 1], 0, [1, 1], batch_norm=None,
                                                         act=act(act_p), name_prefix='enc8'),
                                        )

        elif self.architecture == 'speccnn9l1':  # 6.5 GB (RAM) ; 0.8 GMultAdd  (batch 256)
            # TODO doc
            act = nn.LeakyReLU
            act_p = 0.1  # Activation param
            self.enc_nn = nn.Sequential(convlayer.Conv2D(1, 24, [5, 5], [2, 2], 2, [1, 1], batch_norm=None,
                                                         act=act(act_p), name_prefix='enc1'),
                                        convlayer.Conv2D(24, 48, [4, 4], [2, 2], 2, [1, 1],
                                                         act=act(act_p), name_prefix='enc2'),
                                        convlayer.Conv2D(48, 96, [4, 4], [2, 2], 2, [1, 1],
                                                         act=act(act_p), name_prefix='enc3'),
                                        convlayer.Conv2D(96, 128, [4, 4], [2, 2], 2, [1, 1],
                                                         act=act(act_p), name_prefix='enc4'),
                                        convlayer.Conv2D(128, 128, [4, 4], [2, 2], 2, [1, 1],
                                                         act=act(act_p), name_prefix='enc5'),
                                        convlayer.Conv2D(128, 256, [4, 4], [2, 2], 2, [1, 1],
                                                         act=act(act_p), name_prefix='enc6'),
                                        convlayer.Conv2D(256, 256, [4, 4], [2, 2], 2, [1, 1],
                                                         act=act(act_p), name_prefix='enc7')
                                        )
            if last_layers_to_remove <= 1:
                self.enc_nn.add_module('4x4conv', convlayer.Conv2D(256, 512, [4, 4], [2, 2], 2, [1, 1],
                                                                   act=act(act_p), name_prefix='enc8'))
            if last_layers_to_remove == 0:
                self.enc_nn.add_module('1x1conv', convlayer.Conv2D(512, 1024, [1, 1], [1, 1], 0, [1, 1], batch_norm=None,
                                                                   act=act(act_p), name_prefix='enc9'))

        # TODO try reduce channels from all skip-connection layers (or the model overfits++), dense only (no add res)
        elif self.architecture == 'rescnn':  # 6.9 GB (RAM) ; 0.85 GMultAdd  (batch 256)
            # TODO doc
            act = nn.LeakyReLU
            act_p = 0.1  # Activation param
            self.enc_nn = nn.Sequential(convlayer.Conv2D(1, 32, [5, 5], [2, 2], 2, [1, 1], batch_norm=None,
                                                         act=act(act_p), name_prefix='enc1'),
                                        convlayer.DenseConv2D(32, 48, 96, [4, 4], [2, 2], 2, [1, 1],
                                                              activation=act(act_p), name_prefix='enc23'),
                                        convlayer.DenseConv2D(96, 128, 256, [4, 4], [2, 2], 2, [1, 1],
                                                              activation=act(act_p), name_prefix='enc45'),
                                        convlayer.ResConv2D(256, 256, [4, 4], [2, 2], 2, [1, 1],
                                                            activation=act(act_p), name_prefix='enc67'),
                                        )
            if last_layers_to_remove <= 1:
                self.enc_nn.add_module('4x4conv', convlayer.Conv2D(256, 512, [4, 4], [2, 2], 2, [1, 1],
                                                                   act=act(act_p), name_prefix='enc8'))
            if last_layers_to_remove == 0:
                self.enc_nn.add_module('1x1conv', convlayer.Conv2D(512, 1024, [1, 1], [1, 1], 0, [1, 1], batch_norm=None,
                                                                   act=act(act_p), name_prefix='enc9'))

        else:
            raise NotImplementedError("Architecture '{}' not available".format(self.architecture))

    def forward(self, x_spectrogram):
        return self.enc_nn(x_spectrogram)


if __name__ == "__main__":

    import torchinfo
    input_size = [32, 1, 257, 347]
    enc = SpectrogramEncoder('speccnn8l1', 610, input_size, 0.1)
    _ = torchinfo.summary(enc, input_size=input_size)

    if False:
        # Test: does the dataloader get stuck here as well? Or only in jupyter notebooks?
        # YEP lots of multiprocessing issues with PyTorch DataLoaders....
        # (even pickle could be an issue... )
        from data import dataset
        dexed_dataset = dataset.DexedDataset()

        enc = SpectrogramCNN(architecture='wavenet_baseline')
        dataloader = torch.utils.data.DataLoader(dexed_dataset, batch_size=32, shuffle=False, num_workers=40)
        spectro, params, midi = next(iter(dataloader))
        print("Input spectrogram tensor: {}".format(spectro.size()))
        encoded_spectrogram = enc(spectro)
        _ = torchinfo.summary(enc, input_size=spectro.size())


