import warnings

import torch
import torch.nn as nn
import torchinfo

from model import convlayer
from model.convlayer import ResBlock3Layers, Conv2D
from model.hierarchicalvae import parse_latent_extract_architecture, parse_main_conv_architecture


class SpectrogramEncoder(nn.Module):
    """ Contains a spectrogram-input CNN and some MLP layers, and outputs the mu and logs(var) values"""
    def __init__(self, conv_architecture: str, latent_inference_architecture: str,
                 dim_z: int, deterministic: bool,
                 input_tensor_size, fc_dropout, output_bn=False, output_dropout_p=0.0,
                 deep_features_mix_level=-1):
        """

        :param conv_architecture: String describing the sequential CNN network (applied to each input channel),
            with options (see parse_main_conv_architecture(...) method)
        :param latent_inference_architecture: String describing the network that infers latent values from the sequential
            outputs of the main CNN (with options, see parse_latent_extract_architecture(...) method)
        :param dim_z: Output tensor will be Nminibatch x (2*dim_z)  (means and log variances for each item)
        :param deterministic: If True, this module will output 0.0 variances only.
        :param input_tensor_size:
        :param fc_dropout:
        :param output_bn:
        :param deep_features_mix_level: (applies to multi-channel spectrograms only) negative int, -1 corresponds
            to the last conv layer (usually 1x1 kernel), -2 to the conv layer before, etc.
        """
        super().__init__()
        self.dim_z = dim_z  # Latent-vector size (2*dim_z encoded values - mu and logs sigma 2)
        self.deterministic = deterministic
        self.num_spectrogram_channels = input_tensor_size[1]
        self.conv_architecture = conv_architecture
        self.latent_inference_architecture = latent_inference_architecture
        self.deep_feat_mix_level = deep_features_mix_level  # FIXME don't use if GRU or LSTM
        self.fc_dropout = fc_dropout
        self.conv_arch_name, self.num_cnn_layers, self.conv_arch_args = \
            parse_main_conv_architecture(self.conv_architecture)
        self.latent_arch_name, self.num_latent_layers, self.latent_arch_args = \
            parse_latent_extract_architecture(self.latent_inference_architecture)

        # - - - - - 1) Main CNN encoder (applied once per input spectrogram channel) - - - - -
        # - - - - - - - - - - and optional 2) CNN features mixer - - - - - - - - - -
        self.single_ch_cnn, features_mixer_cnn = self._build_cnns()  # features_mixer_cnn may be deleted/unused

        # - - - - - 3) Latent inference network (from sequential CNN outputs) - - - - -
        self.latent_inference = nn.Sequential()
        # Automatic sequential/recurrent CNN output tensor size inference
        with torch.no_grad():
            self._train_input_tensor_size = input_tensor_size
            single_element_input_tensor_size = list(self._train_input_tensor_size)
            single_element_input_tensor_size[0] = 1  # single-element batch
            dummy_spectrogram = torch.zeros(single_element_input_tensor_size)
            seq_cnn_out, w = self._sequential_cnn(dummy_spectrogram, None)  # FIXME use dummy style
        if self.latent_arch_name == 'mlp':
            # - - - - - 3a) MLP for extracting properly-sized latent vector - - - - -
            mlp = nn.Sequential()
            # Automatic features mixing CNN output tensor size inference
            with torch.no_grad():
                cnn_out, _ = features_mixer_cnn((seq_cnn_out, w))
            cnn_out_size = cnn_out.size()
            cnn_out_items = cnn_out_size[1] * cnn_out_size[2] * cnn_out_size[3]
            # Number of linear layers as configured in the arch arg
            # Default: no final batch-norm (maybe added after this for loop)
            num_hidden_units = 2 * self.dim_z
            num_output_units = self.dim_z if self.deterministic else 2 * self.dim_z
            for i in range(self.num_latent_layers):
                if self.fc_dropout > 0.0:
                    mlp.add_module("encdrop{}".format(i), nn.Dropout(self.fc_dropout))
                in_units = cnn_out_items if (i == 0) else num_hidden_units
                out_units = num_hidden_units if (i < (self.num_latent_layers - 1)) else num_output_units
                mlp.add_module("encfc{}".format(i), nn.Linear(in_units, out_units))
                if i < (self.num_latent_layers - 1):  # No final activation - outputs are latent mu/logvar
                    mlp.add_module("act{}".format(i), nn.ReLU())
            # Batch-norm here to compensate for unregularized z0 of a flow-based latent space (replace 0.1 Dkl)
            # Dropout to help prevent VAE posterior collapse --> should be zero
            if output_bn:
                mlp.add_module('lat_in_regularization', nn.BatchNorm1d(num_output_units))
            if output_dropout_p > 0.0:
                mlp.add_module('lat_in_drop', nn.Dropout(output_dropout_p))
            self.latent_inference = ConvMlpLatentInference(features_mixer_cnn, mlp)
        else:
            # - - - - - 3b) TODO - - - - -
            raise NotImplementedError()  # TODO try LSTM, GRU, transformer....

    def _build_cnns(self):
        """ Builds the main sequential CNN (applied to the input spectrograms, channels are sequence indices)
            and the features mixer CNN, which may not be used. """
        # TODO implement a structure to possibly obtain a hierarchical VAE (multiple outputs from the single_ch_cnn)
        single_ch_cnn, features_mixer_cnn = nn.Sequential(), nn.Sequential()
        if self.conv_arch_name == 'speccnn8l':
            ''' Where to use BN? 'ESRGAN' generator does not use BN in the first and last conv layers.
            DCGAN: no BN on discriminator in out generator out.
            Our experiments seem to show: more stable latent loss with no BN before the FC that regresses mu/logvar,
            consistent training runs  '''
            if self.conv_arch_args['adain']:
                warnings.warn("'adain' arch arg (MIDI notes provided to layers) not implemented")
            if self.conv_arch_args['time+']:
                raise NotImplementedError("_time+ (increased time resolution) arch arg not implemented")
            _in_ch, _out_ch = -1, -1  # backup for res blocks
            building_res_block = False  # if True, the current layer will be added from the next iteration
            finish_res_block = False  # if True, the current layer will include the previous one (res block)
            for i in range(0, self.num_cnn_layers):
                if self.conv_arch_args['res']:
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
                    if self.conv_arch_args['bigger']:
                        out_ch = max(out_ch, 128)
                    elif self.conv_arch_args['big']:
                        out_ch = out_ch if out_ch > 64 else out_ch * 2
                elif 1 <= i <= 6:
                    kernel_size, stride, padding = [4, 4], [2, 2], 2
                    in_ch = 2**(i+2)
                    out_ch = 2**(i+3)
                    if self.conv_arch_args['bigger']:
                        in_ch, out_ch = max(in_ch, 128), max(out_ch, 128)
                    elif self.conv_arch_args['big']:
                        in_ch = in_ch if in_ch > 64 else in_ch * 2
                        out_ch = out_ch if out_ch > 64 else out_ch * 2
                else:  # i == 7
                    kernel_size, stride, padding = [1, 1], [1, 1], 0
                    in_ch = 2**(i+2)
                    out_ch = 2**(i+3)
                # Increased number of layers - sequential encoder
                if self.num_spectrogram_channels > 1:
                    # Does this layer receive the stacked feature maps? (much larger input, 50% larger output)
                    if i == (self.num_cnn_layers + self.deep_feat_mix_level):  # negative mix level
                        in_ch = in_ch * self.num_spectrogram_channels
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
                self_attention = (self.conv_arch_args['att'] and (2 <= (i-1) < (self.num_cnn_layers-2)),
                                  self.conv_arch_args['att'] and (2 <= i < (self.num_cnn_layers-2)))
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
                        single_ch_cnn.add_module(name, conv_layer)
                    else:
                        features_mixer_cnn.add_module(name, conv_layer)

        elif self.conv_arch_name.startswith("sprescnn"):
            raise AssertionError("This arch must be refactored (proper decomposition into seq/non-seq networks)")
            if self.conv_arch_args['res']:
                print("[encoder.py] useless '_res' arch arg for architecture '{}'".format(self.conv_arch_name))
            norm = 'bn+adain' if self.conv_arch_args['adain'] else 'bn'
            # This network is based on a several 'main' blocks. Inside each 'main' block, the resolution is constant.
            # Each 'main' block is made of several res conv blocks. Resolution decreases at the end of each block.
            main_blocks_indices = [0, 1, 2, 3, 4, 5]
            if self.conv_arch_args['big']:
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
                            if main_block_idx >= 4 and self.conv_arch_args['time+']:
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
            self.features_mixer_cnn.add_module('1x1', Conv2D(
                res_blocks_ch[-1], 64 if self.conv_arch_args['time+'] else 128, (1, 1), (1, 1), (0, 0),
                act=nn.Identity(), norm_layer=None))

        else:
            raise AssertionError("Convolutional architecture {} not available".format(self.conv_arch_name))

        return single_ch_cnn, features_mixer_cnn

    def get_single_ch_cnn_summary(self, depth=5):
        """ Return the torchinfo summary of the CNN that is sequentially applied to each input spectrogram. """
        single_ch_input_size = list(self._train_input_tensor_size)
        single_ch_input_size[1] = 1
        return torchinfo.summary(
            self.single_ch_cnn,
            input_data=((torch.zeros(single_ch_input_size), torch.zeros(0)), ),  # empty style tensor, None not accepted
            depth=depth, verbose=0, device=torch.device('cpu'),
            col_names=("input_size", "kernel_size", "output_size", "num_params", "mult_adds"),
            row_settings=("depth", "var_names")
        )

    def get_latent_inference_summary(self, depth=5):
        """ Returns the torchinfo summary of the network which infers latent values from the multiple outputs
        from the single-channel CNN (applied to the sequence of input spectrograms). """
        # FIXME empty style tensor, None not accepted
        with torch.no_grad():
            seq_cnn_out, w = self._sequential_cnn(torch.zeros(self._train_input_tensor_size), torch.zeros(0))
        return torchinfo.summary(
            self.latent_inference, input_data=(seq_cnn_out, w),
            depth=depth, verbose=0, device=torch.device('cpu'),
            col_names=("input_size", "kernel_size", "output_size", "num_params", "mult_adds"),
            row_settings=("depth", "var_names")
        )

    def _sequential_cnn(self, x_spectrograms, w_style):
        # TODO split style (MIDI notes??) before passing it to the single ch CNNs
        # apply main cnn multiple times
        single_channel_cnn_out = [self.single_ch_cnn((torch.unsqueeze(x_spectrograms[:, ch, :, :], dim=1), w_style))
                                  for ch in range(self.num_spectrogram_channels)]
        # Remove w output (sequential module: conditioning passed to all layers)
        single_channel_cnn_out = [x[0] for x in single_channel_cnn_out]
        return torch.cat(single_channel_cnn_out, dim=1), w_style

    def forward(self, x_spectrograms, w_style=None):
        n_minibatch = x_spectrograms.size()[0]
        sequential_cnn_out, _ = self._sequential_cnn(x_spectrograms, w_style)
        # Use features from all input channels and get latent values
        z_mu_logvar = self.latent_inference(sequential_cnn_out, w_style)
        # Last dim contains a latent proba distribution value, last-1 dim is 2 (to retrieve mu or logs sigma2)
        if not self.deterministic:
            return torch.reshape(z_mu_logvar, (n_minibatch, 2, self.dim_z))
        # or: constant log var if this encoder is deterministic (log var is not computed at all)
        else:
            z_mu = torch.unsqueeze(z_mu_logvar, 1)
            z_logvar = torch.ones_like(z_mu) * (- 1e-10)
            return torch.cat([z_mu, z_logvar], 1)

    def set_attention_gamma(self, gamma):
        for mod in self.single_ch_cnn:
            mod.set_attention_gamma(gamma)
        self.latent_inference.set_attention_gamma(gamma)


class ConvMlpLatentInference(nn.Module):
    def __init__(self, conv_features_mixer: nn.Sequential, mlp_module: nn.Sequential):
        super().__init__()
        self.conv_features_mixer = conv_features_mixer
        self.mlp = mlp_module

    def forward(self, x, w):
        cnn_out, _ = self.conv_features_mixer((x, w))
        cnn_out = cnn_out.view(x.shape[0], -1)  # 2nd dim automatically inferred
        return self.mlp(cnn_out)

    def set_attention_gamma(self, gamma):
        for mod in self.conv_features_mixer:
            mod.set_attention_gamma(gamma)


if __name__ == '__main__':  # for debugging
    import config
    import model.build
    model_config, train_config = config.ModelConfig(), config.TrainConfig()
    config.update_dynamic_config_params(model_config, train_config)
    encoder_model, decoder_model, ae_model = model.build.build_ae_model(model_config, train_config)

    summary = encoder_model.get_single_ch_cnn_summary()
    print(summary)

    dummy_input_spec = torch.zeros(model_config.input_audio_tensor_size)
    dummy_z = encoder_model(dummy_input_spec, None)

    #encoder_model.set_attention_gamma(0.7)
