import warnings

import numpy as np
import torch
import torch.nn as nn

from model import convlayer
from model.convlayer import ResBlock3Layers, Conv2D, TConv2D


def parse_architecture(full_architecture: str):
    """ Parses an argument used to describe the encoder and decoder architectures (e.g. speccnn8l1_big_res) """
    # Decompose architecture to retrieve number of cnn and fc layer, options, ...
    arch_args = full_architecture.split('_')
    base_arch_name = arch_args[0]  # type: str
    del arch_args[0]
    if base_arch_name.startswith('speccnn'):
        layers_args = [int(s) for s in base_arch_name.replace('speccnn', '').split('l')]
        num_cnn_layers, num_fc_layers = layers_args[0], layers_args[1]
    elif base_arch_name.startswith('sprescnn'):
        num_cnn_layers, num_fc_layers = None, 1
    else:
        raise AssertionError("Base architecture not available for given arch '{}'".format(base_arch_name))
    # Check arch args, transform
    arch_args_dict = {'adain': False, 'big': False, 'bigger': False, 'res': False, 'att': False, 'time+': False}
    for arch_arg in arch_args:
        if arch_arg in ['adain', 'big', 'bigger', 'res', 'time+', 'att']:
            arch_args_dict[arch_arg] = True  # Authorized arguments
        else:
            raise ValueError("Unvalid encoder argument '{}' from architecture '{}'".format(arch_arg, full_architecture))
    return base_arch_name, num_cnn_layers, num_fc_layers, arch_args_dict


class SpectrogramEncoder(nn.Module):
    """ Contains a spectrogram-input CNN and some MLP layers, and outputs the mu and logs(var) values"""
    def __init__(self, architecture, dim_z, deterministic: bool,
                 input_tensor_size, fc_dropout, output_bn=False, output_dropout_p=0.0,
                 deep_features_mix_level=-1, force_bigger_network=False):
        """

        :param architecture: String describing the type of network, number of CNN and FC layers, with options
            (see parse_architecture(...) method)
        :param dim_z: Output tensor will be Nminibatch x (2*dim_z)  (means and log variances for each item)
        :param deterministic: If True, this module will 0.0 variances only.
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
        self.deterministic = deterministic
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
            self.cnn_out_size = self._forward_cnns(dummy_spectrogram, None).size()  # FIXME use dummy style
        cnn_out_items = self.cnn_out_size[1] * self.cnn_out_size[2] * self.cnn_out_size[3]
        # Number of linear layers as configured in the arch arg (e.g. speccnn8l1 -> 1 FC layer).
        # Default: no final batch-norm (maybe added after this for loop). Always 1024 hidden units
        self.mlp = nn.Sequential()
        module_output_units = self.dim_z if self.deterministic else 2 * self.dim_z
        for i in range(self.num_fc_layers):
            if self.fc_dropout > 0.0:
                self.mlp.add_module("encdrop{}".format(i), nn.Dropout(self.fc_dropout))
            in_units = cnn_out_items if (i == 0) else 1024
            out_units = 1024 if (i < (self.num_fc_layers - 1)) else module_output_units
            self.mlp.add_module("encfc{}".format(i), nn.Linear(in_units, out_units))
            if i < (self.num_fc_layers - 1):  # No final activation - outputs are latent mu/logvar
                self.mlp.add_module("act{}".format(i), nn.ReLU())
        # Batch-norm here to compensate for unregularized z0 of a flow-based latent space (replace 0.1 Dkl)
        # Dropout to help prevent VAE posterior collapse --> should be zero
        if output_bn:
            self.mlp.add_module('lat_in_regularization', nn.BatchNorm1d(module_output_units))
        if output_dropout_p > 0.0:
            self.mlp.add_module('lat_in_drop', nn.Dropout(output_dropout_p))

    def _build_cnns(self):
        if self.base_arch_name == 'speccnn8l1':
            ''' Where to use BN? 'ESRGAN' generator does not use BN in the first and last conv layers.
            DCGAN: no BN on discriminator in out generator out.
            Our experiments seem to show: more stable latent loss with no BN before the FC that regresses mu/logvar,
            consistent training runs  '''
            if self.arch_args['adain']:
                warnings.warn("'adain' arch arg (MIDI notes provided to layers) not implemented")
            if self.arch_args['time+']:
                raise NotImplementedError("_time+ (increased time resolution) arch arg not implemented")
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
                    if self.arch_args['bigger']:
                        out_ch = max(out_ch, 128)
                    elif self.arch_args['big']:
                        out_ch = out_ch if out_ch > 64 else out_ch * 2
                elif 1 <= i <= 6:
                    kernel_size, stride, padding = [4, 4], [2, 2], 2
                    in_ch = 2**(i+2)
                    out_ch = 2**(i+3)
                    if self.arch_args['bigger']:
                        in_ch, out_ch = max(in_ch, 128), max(out_ch, 128)
                    elif self.arch_args['big']:
                        in_ch = in_ch if in_ch > 64 else in_ch * 2
                        out_ch = out_ch if out_ch > 64 else out_ch * 2
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
                # involves matrix multiplications on flattened 1D feature maps -> for smaller 2D feature maps only
                # Also: attention is not useful when convolution kernel size is close to the feature maps' size
                self_attention = (self.arch_args['att'] and (2 <= (i-1) < (self.num_cnn_layers-2)),
                                  self.arch_args['att'] and (2 <= i < (self.num_cnn_layers-2)))
                if building_res_block:
                    _in_ch, _out_ch = in_ch, out_ch
                else:
                    if finish_res_block:
                        name = 'enc_{}_{}'.format(i-1, i)
                        conv_layer = convlayer.ResConv2D(
                            _in_ch, in_ch, out_ch, kernel_size, stride, padding, act=act, norm_layer=norm,
                            adain_num_style_features=None, self_attention=self_attention)
                    else:
                        name = 'enc{}'.format(i)
                        conv_layer = convlayer.Conv2D(
                            in_ch, out_ch, kernel_size, stride, padding, act=act, norm_layer=norm,
                            adain_num_style_features=None, self_attention=self_attention[1])
                    if i < (self.num_cnn_layers + self.deep_feat_mix_level):  # negative mix level
                        self.single_ch_cnn.add_module(name, conv_layer)
                    else:
                        self.features_mixer_cnn.add_module(name, conv_layer)

        elif self.base_arch_name.startswith("sprescnn"):
            if self.arch_args['res']:
                print("[encoder.py] useless '_res' arch arg for architecture '{}'".format(self.base_arch_name))
            norm = 'bn+adain' if self.arch_args['adain'] else 'bn'
            # This network is based on a several 'main' blocks. Inside each 'main' block, the resolution is constant.
            # Each 'main' block is made of several res conv blocks. Resolution decreases at the end of each block.
            main_blocks_indices = [0, 1, 2, 3, 4, 5]
            if self.arch_args['big']:
                res_blocks_counts = [1, 1, 2, 3, 4, 3]
                res_blocks_ch = [64, 128, 128, 256, 512, 1024]
            else:
                res_blocks_counts = [1, 1, 2, 2, 3, 2]
                res_blocks_ch = [8, 32, 128, 256, 512, 1024]
            layer_idx = -1
            self.num_cnn_layers = sum(res_blocks_counts)
            for main_block_idx in main_blocks_indices:
                for res_block_idx in range(res_blocks_counts[main_block_idx]):
                    layer_idx += 1
                    if layer_idx == 0:  # Kernel 7, stride 2, padding 3
                        self.single_ch_cnn.add_module('conv0', Conv2D(1, res_blocks_ch[0], (7, 7), (2, 2), (3, 3),
                                                                      act=nn.Identity(), norm_layer=None))
                    else:  # All other layers: kernel 3, variable stride (through downsample arg) and padding
                        out_ch = res_blocks_ch[main_block_idx]
                        # First block of a main block adapts the size
                        in_ch = res_blocks_ch[main_block_idx - 1] if res_block_idx == 0 else out_ch
                        if res_block_idx > 0:
                            downsample = (False, False)
                        else:
                            if main_block_idx >= 4 and self.arch_args['time+']:
                                downsample = (True, False)  # Option: no temporal stride for last layers
                            else:
                                downsample = (True, True)
                        l = ResBlock3Layers(in_ch, out_ch//4, out_ch, act=nn.LeakyReLU(0.1), downsample=downsample,
                                            norm_layer=norm, adain_num_style_features=None)  # TODO num style features
                        if layer_idx < (self.num_cnn_layers + self.deep_feat_mix_level):  # negative mix level
                            self.single_ch_cnn.add_module('resblk{}'.format(layer_idx), l)
                        else:
                            self.features_mixer_cnn.add_module('resblk{}'.format(layer_idx), l)
            # A final 1x1 must be used to reduce the huge number of channels (w/ large feat maps) before the FC layer
            self.features_mixer_cnn.add_module('1x1', Conv2D(res_blocks_ch[-1], 64 if self.arch_args['time+'] else 128,
                                                             (1, 1), (1, 1), (0, 0),
                                                             act=nn.Identity(), norm_layer=None))

        else:
            raise AssertionError("Architecture {} not available".format(self.base_arch_name))

    def _forward_cnns(self, x_spectrograms, w_style):
        # TODO split style (MIDI notes??) before passing it to the single ch CNNs
        # apply main cnn multiple times
        single_channel_cnn_out = [self.single_ch_cnn((torch.unsqueeze(x_spectrograms[:, ch, :, :], dim=1), w_style))
                                  for ch in range(self.spectrogram_channels)]
        # Remove w output (sequential module: conditioning passed to all layers)
        single_channel_cnn_out = [x[0] for x in single_channel_cnn_out]
        # Then mix features from different input channels
        x_out, w = self.features_mixer_cnn((torch.cat(single_channel_cnn_out, dim=1), w_style))
        return x_out

    def forward(self, x_spectrograms, w_style=None):
        n_minibatch = x_spectrograms.size()[0]
        cnn_out = self._forward_cnns(x_spectrograms, w_style).view(n_minibatch, -1)  # 2nd dim automatically inferred
        # print("Forward CNN out size = {}".format(cnn_out.size()))
        z_mu_logvar = self.mlp(cnn_out)
        # Last dim contains a latent proba distribution value, last-1 dim is 2 (to retrieve mu or logs sigma2)
        if not self.deterministic:
            return torch.reshape(z_mu_logvar, (n_minibatch, 2, self.dim_z))
        # or: constant log var if this encoder is deterministic (log var is not computed at all)
        else:
            z_mu = torch.unsqueeze(z_mu_logvar, 1)
            z_logvar = torch.ones_like(z_mu) * (- 1e-10)
            return torch.cat([z_mu, z_logvar], 1)

    def set_attention_gamma(self, gamma):
        for mod in self.features_mixer_cnn:
            mod.set_attention_gamma(gamma)
        for mod in self.single_ch_cnn:
            mod.set_attention_gamma(gamma)


if __name__ == '__main__':  # for debugging
    import config
    config.update_dynamic_config_params()
    import model.build
    encoder_model, decoder_model, ae_model = model.build.build_ae_model(config.model, config.train)
    encoder_model.set_attention_gamma(0.7)
