
import unittest

import torch
import torch.nn as nn

from model import convlayer



class DummyModel(nn.Module):
    def __init__(self, res_block: convlayer.ResBlock3Layers):
        super().__init__()
        self.res_block = res_block

    def forward(self, x):
        return self.res_block((x, torch.zeros((1, 3))))  # Fixed set of 3 AdaIN features

    def get_x_and_res_before_add(self, x):
        res, _ = self.res_block.res_convs((x, torch.zeros((1, 3))))
        if self.res_block.resize_module is not None:
            return self.res_block.resize_module(x), res
        else:
            return x, res


class TestResBlock3Layers(unittest.TestCase):

    def test_norm_layers(self):
        for n in ['bn', 'adain', 'bn+adain', None]:
            res_block = convlayer.ResBlock3Layers(64, 16, 64, norm_layer=n, adain_num_style_features=3)
            dummy_m = DummyModel(res_block)
            for input_size in [(160, 64, 2, 2), (1, 64, 10, 10), (1, 64, 11, 11)]:
                x_input = torch.zeros(input_size)
                x_output, w = dummy_m(x_input)
                self.assertEqual(x_input.shape, x_output.shape)

    def test_padding(self):
        for i in range(10):
            for j in range(10):
                res_block = convlayer.ResBlock3Layers(64, 16, 64,
                                                      upsample=(False, False), downsample=(False, False),
                                                      extra_padding=(i, j),
                                                      norm_layer='bn+adain', adain_num_style_features=3)
                dummy_m = DummyModel(res_block)
                for input_size in [(160, 64, 2, 2), (1, 64, 10, 10), (1, 64, 11, 11)]:
                    x_input = torch.zeros(input_size)
                    x_out, res = dummy_m.get_x_and_res_before_add(x_input)
                    self.assertEqual(x_out.shape, res.shape)

    def test_upsampling(self):
        for upsample in [(False, False), (True, False), (False, True), (True, True)]:
            for extra_padding in [(0, 0), (1, 0), (0, 1), (1, 1), (2, 2), (3, 3)]:
                res_block = convlayer.ResBlock3Layers(64, 16, 64,
                                                      upsample=upsample, downsample=(False, False),
                                                      extra_padding=extra_padding,
                                                      norm_layer='bn+adain', adain_num_style_features=3)
                dummy_m = DummyModel(res_block)
                for input_size in [(160, 64, 2, 2), (160, 64, 3, 3), (1, 64, 10, 10), (1, 64, 11, 11)]:
                    x_input = torch.zeros(input_size)
                    x_out, res = dummy_m.get_x_and_res_before_add(x_input)
                    self.assertEqual(x_out.shape, res.shape)

    def test_downsampling(self):
        for downsample in [(False, False), (True, False), (False, True), (True, True)]:
            for extra_padding in [(0, 0), (1, 0), (0, 1), (1, 1), (2, 2), (3, 3)]:
                res_block = convlayer.ResBlock3Layers(64, 16, 64,
                                                      upsample=(False, False), downsample=downsample,
                                                      extra_padding=extra_padding,
                                                      norm_layer='bn+adain', adain_num_style_features=3)
                dummy_m = DummyModel(res_block)
                for input_size in [(160, 64, 2, 2), (160, 64, 3, 3), (1, 64, 10, 10), (1, 64, 11, 11)]:
                    x_input = torch.zeros(input_size)
                    x_out, res = dummy_m.get_x_and_res_before_add(x_input)
                    self.assertEqual(x_out.shape, res.shape)

    @staticmethod
    def _try_get_output(in_ch, middle_ch, out_ch, upsample, downsample):
        for extra_padding in [(0, 0), (1, 0), (0, 1), (1, 1), (2, 2), (3, 3)]:
            res_block = convlayer.ResBlock3Layers(in_ch, middle_ch, out_ch,
                                                  upsample=upsample, downsample=downsample,
                                                  extra_padding=extra_padding,
                                                  norm_layer='bn+adain', adain_num_style_features=3)
            dummy_m = DummyModel(res_block)
            for input_size in [(160, in_ch, 2, 2), (160, in_ch, 3, 3), (1, in_ch, 10, 10), (1, in_ch, 11, 11)]:
                x_input = torch.zeros(input_size)
                dummy_m(x_input)

    def test_num_channels(self):
        for upsample in [(False, False), (True, False), (False, True), (True, True)]:
            downsample = (False, False)
            for in_ch in [47, 64]:
                for out_ch in [25, 32, 47, 64]:
                    for middle_ch_ratio in [2, 4]:
                        self._try_get_output(in_ch, out_ch//middle_ch_ratio, out_ch, upsample, downsample)
        for downsample in [(False, False), (True, False), (False, True), (True, True)]:
            upsample = (False, False)
            for in_ch in [47, 64]:
                for out_ch in [47, 64, 93, 128]:
                    for middle_ch_ratio in [2, 4]:
                        self._try_get_output(in_ch, out_ch//middle_ch_ratio, out_ch, upsample, downsample)


if __name__ == "__main__":
    unittest.main()
    # TODO tests :
    #  - stride, up sampling

