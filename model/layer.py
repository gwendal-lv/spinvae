
"""
Defines some basic layer Classes to be integrated into bigger networks
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Conv2D(nn.Sequential):
    """ A basic conv layer with activation and batch-norm """
    def __init__(self, in_ch, out_ch, kernel_size, stride, padding, dilation=(1, 1),
                 padding_mode='zeros', act=nn.ReLU(), name_prefix='', bn='after'):
        """

        :param bn: 'after' activation, 'before' activation, or None
        """
        super().__init__()
        self.add_module(name_prefix + 'conv', nn.Conv2d(in_ch, out_ch, kernel_size, stride,
                                                        padding, dilation, padding_mode=padding_mode))
        if bn == 'before':
            self.add_module(name_prefix + 'bn', nn.BatchNorm2d(out_ch))
        self.add_module(name_prefix + 'act', act)
        if bn == 'after':
            self.add_module(name_prefix + 'bn', nn.BatchNorm2d(out_ch))


class TConv2D(nn.Sequential):
    """ A basic Transposed conv layer with activation and batch-norm """
    def __init__(self, in_ch, out_ch, kernel_size, stride, padding, output_padding=0, dilation=1,
                 padding_mode='zeros', act=nn.ReLU(), name_prefix='', bn='after'):
        """

        :param bn: 'after' activation, 'before' activation, or None
        """
        super().__init__()
        self.add_module(name_prefix + 'tconv', nn.ConvTranspose2d(in_ch, out_ch, kernel_size, stride,
                                                                  padding, output_padding,
                                                                  dilation=dilation, padding_mode=padding_mode))
        if bn == 'before':
            self.add_module(name_prefix + 'bn', nn.BatchNorm2d(out_ch))
        self.add_module(name_prefix + 'act', act)
        if bn == 'after':
            self.add_module(name_prefix + 'bn', nn.BatchNorm2d(out_ch))


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
        self.conv1 = Conv2D(in_ch, hidden_ch, kernel_size, stride, padding, dilation,
                            padding_mode, activation, name_prefix, batch_norm)
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



class ResConv2D(nn.Module):
    """ 2 convolutional layers with input added to the output. Out ch will be equal to in ch count.
    An automatic average pooling is performed if the conv are strided are reduce the size of the feature maps.

    Activation and BN are performed after the add operation. """
    def __init__(self, in_ch, hidden_ch,
                 kernel_size, stride, padding, dilation, padding_mode='zeros',
                 activation=nn.ReLU(), name_prefix='', batch_norm='after',
                 pool_type='avg'):
        """

        :param batch_norm: 'after' activation, 'before' activation, or None
        """
        super().__init__()
        out_ch = in_ch
        self.conv1_act_bn = Conv2D(in_ch, hidden_ch, kernel_size, stride, padding, dilation,
                                   padding_mode, activation, name_prefix, batch_norm)
        self.conv2 = nn.Conv2d(hidden_ch, out_ch, kernel_size, stride, padding, dilation, padding_mode=padding_mode)
        # TODO sequence, choose BN/act ordering
        self.act2 = activation
        self.bn2 = nn.BatchNorm2d(out_ch)

    def forward(self, x):
        x_hidden = self.conv1_act_bn(x)
        x_before_add = self.conv2(x_hidden)
        output_feature_maps_size = x_before_add.shape[2:4]
        # Residuals obtained from automatic average pool
        return self.bn2(self.act2(x_before_add + F.adaptive_avg_pool2d(x, output_feature_maps_size)))


class ResTConv2D(nn.Module):
    """ 2 transpose convolution layers with input added to the output. Out ch will be equal to in ch count.
    An automatic upsampling bilinear interpolation is performed to increase the size of the feature maps.

    Activation and BN are performed after the add operation. """
    def __init__(self, in_ch, hidden_ch,
                 kernel_size, stride, padding, output_padding, dilation=1, padding_mode='zeros',
                 activation=nn.ReLU(), name_prefix='', batch_norm='after',
                 pool_type='avg'):
        """

        :param batch_norm: 'after' activation, 'before' activation, or None
        """
        super().__init__()
        out_ch = in_ch
        self.conv1_act_bn = TConv2D(in_ch, hidden_ch, kernel_size, stride, padding, output_padding, dilation,
                                    padding_mode, activation, name_prefix, batch_norm)
        self.conv2 = nn.ConvTranspose2d(in_ch, out_ch, kernel_size, stride, padding, output_padding,
                                        dilation=dilation, padding_mode=padding_mode)
        # TODO sequence, choose BN/act ordering
        self.act2 = activation
        self.bn2 = nn.BatchNorm2d(out_ch)

    def forward(self, x):
        x_hidden = self.conv1_act_bn(x)
        x_before_add = self.conv2(x_hidden)
        output_feature_maps_size = x_before_add.shape[2:4]
        # Residuals obtained from automatic average pool
        return self.bn2(self.act2(x_before_add +
                                  F.interpolate(x, output_feature_maps_size, mode='bilinear', align_corners=False)))

