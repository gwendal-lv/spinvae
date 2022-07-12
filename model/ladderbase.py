"""
Base classes and methods for a ladder encoder or decoder.
"""
from abc import abstractmethod
from typing import Tuple

import torch
import torch.nn as nn


def parse_main_conv_architecture(full_architecture: str):
    """ Parses an argument used to describe the encoder and decoder conv architectures (e.g. speccnn8l_big_res) """
    # Decompose architecture to retrieve number of conv layers, options, ...
    arch_args = full_architecture.split('_')
    base_arch_name = arch_args[0]  # type: str
    del arch_args[0]
    if base_arch_name.startswith('speccnn'):
        n_blocks = int(base_arch_name.replace('speccnn', '').replace('l', ''))
        n_layers_per_block = 1
    elif base_arch_name.startswith('sprescnn'):
        n_blocks = None
        n_layers_per_block = None
    elif base_arch_name.startswith('specladder'):
        blocks_args = base_arch_name.replace('specladder', '').split('x')
        n_blocks, n_layers_per_block = int(blocks_args[0]), int(blocks_args[1])
    else:
        raise AssertionError("Base architecture not available for given arch '{}'".format(base_arch_name))
    # Check arch args, transform
    arch_args_dict = {'adain': False, 'big': False, 'bigger': False, 'res': False, 'att': False,
                      'depsep5x5': False, 'swish': False, 'wn': False}
    for arch_arg in arch_args:
        if arch_arg in arch_args_dict.keys():
            arch_args_dict[arch_arg] = True  # Authorized arguments
        else:
            raise ValueError("Unvalid encoder argument '{}' from architecture '{}'".format(arch_arg, full_architecture))
    return {'name': base_arch_name, 'n_blocks': n_blocks, 'n_layers_per_block': n_layers_per_block,
            'args': arch_args_dict}


class LadderBase(nn.Module):
    def __init__(self, conv_arch, latent_arch):
        super().__init__()
        self.single_ch_conv_arch = conv_arch
        self.latent_arch = latent_arch

    def _compute_next_H_W(self, input_H_W: Tuple[int, int], strided_conv_block: nn.Module):
        with torch.no_grad():
            x = torch.zeros((1, strided_conv_block.in_channels) + input_H_W)  # Tuple concat
            return tuple(strided_conv_block(x).shape[2:])

    def _get_conv_act(self):
        if self.single_ch_conv_arch['args']['swish']:
            return nn.SiLU()
        else:
            return nn.LeakyReLU(0.1)

    def _get_conv_norm(self):
        if self.single_ch_conv_arch['args']['wn']:
            return 'wn'
        else:  # Default (no arg): Batch Norm
            return 'bn'

    @abstractmethod
    def get_custom_group_module(self, group_name: str) -> nn.Module:
        """
        Returns a module of parameters corresponding to a given group (e.g. 'audio', 'latent', ...).
        That means that even if a group is split into different Modules, they have to be stored in,
        e.g., a single ModuleList.
        """
        pass
