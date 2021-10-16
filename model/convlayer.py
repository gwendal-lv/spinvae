
"""
Defines some basic layer Classes to be integrated into bigger networks
"""
import copy
from typing import Optional
from abc import ABC, abstractmethod

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F



class AdaIN(nn.Module):
    def __init__(self, num_style_features, num_conv_ch):
        """ Adaptive Instance Normalization, inspired by the StyleGAN generator architecture. """
        super().__init__()
        self.norm = nn.InstanceNorm2d(num_conv_ch, affine=False)  # No learnable params
        # Affine transforms (through basic affine FC layers) to compute normalization bias and scale
        self.bias_nn = nn.Linear(num_style_features, num_conv_ch)
        self.scale_nn = nn.Linear(num_style_features, num_conv_ch)

    def forward(self, x, w_style):
        """ Normalizes and rescales/re-biases the 2D feature maps x using style vectors w. """
        x = self.norm(x)
        b, s = self.bias_nn(w_style), self.scale_nn(w_style)
        # x shape : N x C x H x W  (W: time ; H: frequency)
        # w shape : N x C
        b = torch.unsqueeze(torch.unsqueeze(b, 2), 3).expand(-1, -1, x.shape[2], x.shape[3])
        s = torch.unsqueeze(torch.unsqueeze(s, 2), 3).expand(-1, -1, x.shape[2], x.shape[3])
        return x * s + b


class ActAndNorm(nn.Module):
    def __init__(self, in_ch, out_ch, act=nn.ReLU(),
                 norm_layer: Optional[str] = 'bn', adain_num_style_features: Optional[int] = None, reverse_order=False):
        """
        Base class for any conv layer with act and norm (usable a base class for Conv or Transpose Conv layers,
        or as main class). The forward function returns a batch of features maps, and a batch of w conditioning vectors.

        :param norm_layer: 'bn' (Batch Norm), 'adain' (Adaptive Instance Norm, requires w_style during forward) or None
        :param reverse_order: Ff False, conv->act->norm. If True, norm->act->conv (res blocks)
        """
        super().__init__()
        self.reverse_order = reverse_order
        self.conv = None  # can be assigned by child class - otherwise, not applied
        self.act = act
        self.norm_layer_description = norm_layer
        if self.norm_layer_description is None:
            self.norm = None
        elif self.norm_layer_description == 'bn':
            self.norm = nn.BatchNorm2d(out_ch if not reverse_order else in_ch)
        elif self.norm_layer_description == 'adain':
            self.norm = AdaIN(adain_num_style_features, out_ch if not reverse_order else in_ch)
        else:
            raise AssertionError("Layer norm {} not available.".format(self.norm_layer_description))

    def forward(self, x_and_w_style):
        """ Always returns a tuple (x, w) for the style to be passed to the next layer in the sequence. """
        x, w_style = x_and_w_style[0], x_and_w_style[1]
        if not self.reverse_order:  # Usual order: conv->act->norm
            if self.conv is not None:
                x = self.conv(x)
            x = self.act(x)
        if self.norm_layer_description == 'bn':
            x = self.norm(x)
        elif self.norm_layer_description == 'adain':
            if w_style is None:
                raise AssertionError("w_style cannot be None when using AdaIN")
            x = self.norm(x, w_style)
        else:
            pass
        if self.reverse_order:  # reversed order: norm->act->conv (used in res blocks)
            x = self.act(x)
            if self.conv is not None:
                x = self.conv(x)
        return x, w_style


class Conv2D(ActAndNorm):
    """ A basic conv layer with activation and normalization layers """
    def __init__(self, in_ch, out_ch, kernel_size, stride, padding, dilation=(1, 1),
                 padding_mode='zeros', act=nn.ReLU(),
                 norm_layer: Optional[str] = 'bn', adain_num_style_features: Optional[int] = None, reverse_order=False):
        super().__init__(in_ch, out_ch, act, norm_layer, adain_num_style_features, reverse_order)
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding, dilation, padding_mode=padding_mode)


class TConv2D(ActAndNorm):
    """ A basic Transposed conv layer with activation and normalization layers """
    def __init__(self, in_ch, out_ch, kernel_size, stride, padding, output_padding=0, dilation=1, padding_mode='zeros',
                 act=nn.ReLU(),
                 norm_layer: Optional[str] = 'bn', adain_num_style_features: Optional[int] = None, reverse_order=False):
        super().__init__(in_ch, out_ch, act, norm_layer, adain_num_style_features, reverse_order)
        self.conv = nn.ConvTranspose2d(in_ch, out_ch, kernel_size, stride, padding, output_padding,
                                       dilation=dilation, padding_mode=padding_mode)



class DenseConv2D(nn.Module):
    """ 2 convolutional layers with input concatenated to the output (and downsampled if strided convs).
    All input channels are stacked at the end (no activation is applied to concatenated inputs). """
    def __init__(self, in_ch, hidden_ch, out_ch,
                 kernel_size, stride, padding, dilation, padding_mode='zeros',
                 activation=nn.ReLU(), name_prefix='', batch_norm='after',
                 pool_type='avg'):
        """

        :param batch_norm: 'after' activation, 'before' activation, or None
        """
        super().__init__()
        raise AssertionError("deprecated")
        self.conv1 = Conv2D(in_ch, hidden_ch, kernel_size, stride, padding, dilation,
                            padding_mode, activation, batch_norm)
        pre_concat_num_ch = out_ch - in_ch
        self.conv2 = Conv2D(hidden_ch, pre_concat_num_ch, kernel_size, stride, padding, dilation,
                            padding_mode, activation, name_prefix, batch_norm)

    def forward(self, x):
        x_hidden = self.conv1(x)
        x_before_concat = self.conv2(x_hidden)
        output_feature_maps_size = x_before_concat.shape[2:4]
        # Residuals obtained from automatic average pool
        return torch.cat((x_before_concat, F.adaptive_avg_pool2d(x, output_feature_maps_size)), dim=1)


class DenseTConv2D(nn.Module):
    """ 2 transpose convolution layers with input upsampled (bilinear interp) and concatenated to the output """
    def __init__(self, in_ch, hidden_ch, res_ch, out_ch,
                 kernel_size, stride, padding, output_padding, dilation=1, padding_mode='zeros',
                 activation=nn.ReLU(), name_prefix='', batch_norm='after'):
        """

        :param res_ch: Number of residual (skip-connection) channels after the 1x1 conv applied to the input.
            Should be (much) less than the number of output channels
        :param batch_norm: 'after' activation, 'before' activation, or None
        """
        super().__init__()
        raise AssertionError("deprecated")
        self.conv1_act_bn = TConv2D(in_ch, hidden_ch, kernel_size, stride, padding, output_padding, dilation,
                                    padding_mode, activation, name_prefix, batch_norm)
        pre_concat_num_ch = out_ch - res_ch
        self.conv2 = nn.ConvTranspose2d(hidden_ch, pre_concat_num_ch, kernel_size, stride, padding, output_padding,
                                        dilation=dilation, padding_mode=padding_mode)
        self.skip_conv_1x1 = nn.Conv2d(in_ch, res_ch, kernel_size=(1, 1))
        # TODO sequence, choose BN/act ordering
        self.act2 = activation
        self.bn2 = nn.BatchNorm2d(out_ch)

    def forward(self, x):
        x_hidden = self.conv1_act_bn(x)
        x_before_concat = self.conv2(x_hidden)
        output_feature_maps_size = x_before_concat.shape[2:4]
        # 1x1 conv to reduce nb of channels to be concatenated
        x_res = self.skip_conv_1x1(x)
        # Upsampled residuals obtained from bilinear interpolation
        x_cat = torch.cat((x_before_concat,
                           F.interpolate(x_res, output_feature_maps_size, mode='bilinear', align_corners=False)), dim=1)
        return self.bn2(self.act2(x_cat))



class ResConv2DBase(nn.Module, ABC):
    """ Base class for ResConv2D and ResTConv2D: 2 convolutional layers with input added to the output
    (if necessary: number of channels adapted using a 1x1 conv).
    Final activation and norm are performed after the add operation.
    """
    def __init__(self, in_ch, out_ch, act=nn.ReLU(),
                 norm_layer: Optional[str] = 'bn', adain_num_style_features: Optional[int] = None):
        super().__init__()
        self.conv1_act_bn = None  # to be set by child class
        self.conv2 = None
        self.residuals_conv = nn.Conv2d(in_ch, out_ch, (1, 1)) if (in_ch != out_ch) else None
        self.act_norm_2 = ActAndNorm(in_ch, out_ch, act, norm_layer, adain_num_style_features)

    @abstractmethod
    def _resize_x_skip(self, x_skip, output_feature_maps_size):
        pass

    def forward(self, x_and_w):
        x, w = x_and_w[0], x_and_w[1]  # conditioning w is always passed with x
        x_hidden, _ = self.conv1_act_bn((x, w))
        x_before_add = self.conv2(x_hidden)
        output_feature_maps_size = x_before_add.shape[2:4]
        # Skip connection - conv, automatic average pool if necessary
        x_skip = self.residuals_conv(x) if self.residuals_conv is not None else x
        if x_skip.shape[2:4] != output_feature_maps_size:
            x_skip = self._resize_x_skip(x_skip, output_feature_maps_size)
        return self.act_norm_2((x_before_add + x_skip, w))


class ResConv2D(ResConv2DBase):
    """ An automatic average pooling is performed on the residuals if necessary,
    in order to reduce the size of the feature maps."""
    def __init__(self, in_ch, hidden_ch, out_ch,
                 kernel_size, stride, padding, dilation=(1, 1), padding_mode='zeros', act=nn.ReLU(),
                 norm_layer: Optional[str] = 'bn', adain_num_style_features: Optional[int] = None):
        super().__init__(in_ch, out_ch, act, norm_layer, adain_num_style_features)
        self.conv1_act_bn = Conv2D(in_ch, hidden_ch, kernel_size, stride, padding, dilation,
                                   padding_mode, act, norm_layer, adain_num_style_features)
        self.conv2 = nn.Conv2d(hidden_ch, out_ch, kernel_size, stride, padding, dilation, padding_mode=padding_mode)

    def _resize_x_skip(self, x_skip, output_feature_maps_size):
        return F.adaptive_avg_pool2d(x_skip, output_feature_maps_size)


class ResTConv2D(ResConv2DBase):
    """  An automatic upsampling bilinear interpolation is performed to increase the size of the feature maps. """
    def __init__(self, in_ch, hidden_ch, out_ch,
                 kernel_size, stride, padding, output_padding, dilation=(1, 1), padding_mode='zeros', act=nn.ReLU(),
                 norm_layer: Optional[str] = 'bn', adain_num_style_features: Optional[int] = None):
        super().__init__(in_ch, out_ch, act, norm_layer, adain_num_style_features)
        self.conv1_act_bn = TConv2D(in_ch, hidden_ch, kernel_size, stride, padding, output_padding, dilation,
                                    padding_mode, act, norm_layer, adain_num_style_features)
        self.conv2 = nn.ConvTranspose2d(hidden_ch, out_ch, kernel_size, stride, padding, output_padding,
                                        dilation=dilation, padding_mode=padding_mode)

    def _resize_x_skip(self, x_skip, output_feature_maps_size):
        return F.interpolate(x_skip, output_feature_maps_size, mode='bilinear', align_corners=False)



class ResBlock3Layers(nn.Module):
    def __init__(self, in_ch, internal_3x3_ch, out_ch, internal_stride=(1, 1), act=nn.ReLU(),
                 norm_layer: Optional[str] = 'bn', adain_num_style_features: Optional[int] = None):
        """
        The reduced number of channels to perform the actual 3x3 conv is inspired by ResNeXt.

        :param norm_layer: If 'bn' or 'adain', is applied to all norm layers. If 'bn+adain', the middle residual
            layers uses AdaIN, the 2 others use BN.
        """
        super().__init__()
        self.in_ch, self.out_ch = in_ch, out_ch
        if norm_layer is None:
            bn, adain = False, False
        elif norm_layer.lower() == 'bn':
            bn, adain = True, False
        elif norm_layer.lower() == 'adain':
            bn, adain = False, True
        elif norm_layer.lower() == 'bn+adain' or norm_layer.lower() == 'adain+bn':
            bn, adain = True, True
        else:
            raise ValueError("'{}' norm layer: unrecognized string argument.".format(norm_layer))
        if adain and adain_num_style_features is None:
            raise AssertionError("AdaIN requires adain_num_style_features not to be None.")
        # Residual network: reduces the nb of channels, applies a 3x3 conv, then increases the nb of ch before add
        # TODO adapter taille residuals si strided conv (taille feature maps diminue)
        # TODO adapter nb ch residuals si nb channels diminue
        #    si nb channels augmente: ajouter résidus sur une partie seulement des ch?
        self.res_convs = nn.Sequential()
        # allow a mix of adain and bn --> 1 adain first, all the rest is BN
        norm_name = 'bn' if bn else ('adain' if adain else None)  # 1st layer: BN has the priority over AdaIN
        self.res_convs.add_module('redu1x1',
                                  Conv2D(in_ch, internal_3x3_ch, (1, 1), (1, 1), (0, 0),  # kernel, stride, padding
                                         act=copy.deepcopy(act), reverse_order=True,
                                         norm_layer=norm_name, adain_num_style_features=adain_num_style_features))

        norm_name = 'adain' if adain else ('bn' if bn else None)  # 2nd layer: AdaIN has the priority over BN
        # Strided convolution (downsampling)
        # TODO bicubic upsampling (3x3 tconv leads to artifacts: https://distill.pub/2016/deconv-checkerboard/)
        # TODO allow custom input padding (to increase size more)
        self.res_convs.add_module('conv3x3',
                                  Conv2D(internal_3x3_ch, internal_3x3_ch, (3, 3), internal_stride, (1, 1),
                                         act=copy.deepcopy(act), reverse_order=True,
                                         norm_layer=norm_name, adain_num_style_features=adain_num_style_features))

        norm_name = 'bn' if bn else ('adain' if adain else None)  # 3rd layer: BN has the priority over AdaIN
        self.res_convs.add_module('incr1x1',
                                  Conv2D(internal_3x3_ch, out_ch, (1, 1), (1, 1), (0, 0),
                                         act=copy.deepcopy(act), reverse_order=True,
                                         norm_layer=norm_name, adain_num_style_features=adain_num_style_features))

    def forward(self, x_and_w):
        """
        :param x_and_w: Tuple: Feature maps x, conditiong (style) vector w
        :return: Output feature maps, conditiong (style) vector w
        """
        x, w = x_and_w[0], x_and_w[1]
        res, _ = self.res_convs(x_and_w)

        # if upsampling AND decreasing ch count: we'll use the method inspired by "the devil is in the decoder"
        #  (https://arxiv.org/abs/1707.05847) -> bilinear (or bicubic) upsampling and add some channels (non-learnable)
        # Upsampling and decreasing ch count should happen simultaneously, but this is not a constraint

        # upsampling or downsampling

        # Addition of residuals (number of in/out channels can be different)
        if self.in_ch == self.out_ch:
            x = x + res
        elif self.in_ch < self.out_ch:  # Increasing number of channels (e.g. in an encoder)
            # The new channels (non-existent in original input x) will include residual values only
            res[:, 0:self.in_ch, :, :] = res[:, 0:self.in_ch, :, :] + x
            x = res
        elif self.in_ch > self.out_ch:  # Decreasing number of channels (e.g. in a decoder)
            # Split input channels into out_ch groups, add (and normalize) all channels from each group
            in_ch_split_indices = np.array_split(np.arange(0, self.in_ch), self.out_ch)
            for out_ch_index, in_ch_indices in enumerate(in_ch_split_indices):
                for in_ch_index in list(in_ch_indices):
                    res[:, out_ch_index, :, :] += (x[:, in_ch_index, :, :] / in_ch_indices.shape[0])
            x = res

        return x, w
