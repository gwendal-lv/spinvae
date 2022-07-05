"""
General classes and methods for modelling presets (can be used by preset encoders and decoders, or other modules).
"""

import torch
import torch.nn as nn

from data.preset2d import Preset2dHelper


def parse_preset_model_architecture(full_architecture: str):
    arch_args = full_architecture.split('_')
    base_arch_name = arch_args[0].lower()
    num_layers = int(arch_args[1].replace('l', ''))
    arch_args_dict = {
         # FIXME ALLOW CONV kernel sizes??? large and/or dilated convs (wavenet-like, but for a sequence of synth pars)
        'posenc': False,  # Positional encodings can be added inside some architectures
        'bn': False,  # Batch-Norm
        'ln': False,  # Layer-Norm
        'elu': False,
        'swish': False,
        # 'gated': False,  # (Self-)gating ("light attention") mechanisms can be added to some architectures
        # 'att': False,  # SAGAN-like self-attention
    }
    for arch_arg in arch_args[2:]:
        if arch_arg in arch_args_dict.keys():
            arch_args_dict[arch_arg] = True  # Authorized arguments
        else:
            raise ValueError("Unvalid encoder argument '{}' from architecture '{}'".format(arch_arg, full_architecture))
    return {'name': base_arch_name, 'n_layers': num_layers, 'args': arch_args_dict}


def get_act(arch_args):
    assert not (arch_args['elu'] and arch_args['swish']), "Can't use 2 activations at once."
    if arch_args['elu']:
        return nn.ELU()
    elif arch_args['swish']:
        return nn.SiLU()
    else:
        return nn.LeakyReLU(0.1)


class PresetEmbedding(nn.Module):
    def __init__(self, hidden_size: int, preset_helper: Preset2dHelper):
        super().__init__()
        self.hidden_size, self.preset_helper = hidden_size, preset_helper

        # Currently no max norm is implemented (categorical or "numerical" embeddings)
        #     We rely on weight decay (and layer norm?) to prevent exploding values

        # TODO is type numerical - to be able to retrieve the embedding inside forward_single_step(...)

        # TODO ctor arg to use "naive basic" transforms, identical for all synth params (interesting experiment)

        # Numerical: use a 1d conv with a lot of output channels
        n_numerical_conv_ch = hidden_size * preset_helper.n_param_types
        # FIXME this method uses very similar embeddings for different discrete numerical synth params
        self.numerical_embedding_conv = nn.Conv1d(1, n_numerical_conv_ch, 1)
        # This mask will have to be expanded (batch dim to the minibatch size) before being used.
        #  Mask broadcasting behaves in a weird way with pytorch 1.10. It's applied to the first dimensions, not
        #  to the last (trailing) dimensions as we could expect https://pytorch.org/docs/stable/notes/broadcasting.html
        self.numerical_embedding_mask = torch.zeros(    # Channels last, after transpose
            (1, self._n_num_params, n_numerical_conv_ch), dtype=torch.bool
        )
        for num_idx, matrix_row in enumerate(preset_helper.matrix_numerical_rows):
            type_class = int(preset_helper.param_types_tensor[matrix_row].item())
            self.numerical_embedding_mask[0, num_idx, (type_class*hidden_size):((type_class+1) * hidden_size)] = True

        # Categorical: Total amount of possible different categorical indexes (for all parameters):
        #       max_categorical_cardinal * n_param_types
        # Most embeddings values will never be used, but the embedding index can be obtained very easily using
        # a simple product (synth param type * class index)
        n_categorical_embeddings = preset_helper.max_cat_classes * preset_helper.n_param_types
        self.categorical_embedding = nn.Embedding(n_categorical_embeddings, hidden_size, max_norm=None)

        a = 0

    @property
    def _n_num_params(self):
        return self.preset_helper.n_learnable_numerical_params

    def forward(self, u_in: torch.Tensor):
        """ Returns an embedding (shape N x L x hidden_size) for an entire input preset (shape N x L x 3) """
        # TODO move masks to device only once
        embed_out = torch.empty((u_in.shape[0], u_in.shape[1], self.hidden_size), device=u_in.device)
        # Categorical: Class values in "column" 0, types in "column" 2 ("column" is the last dimension)
        u_categorical = u_in[:, self.preset_helper.matrix_categorical_bool_mask, :]
        u_cat_embed_idx = torch.round(
            u_categorical[:, :, 2] * self.preset_helper.max_cat_classes + u_categorical[:, :, 0]
        ).long()
        u_categorical_embeds = self.categorical_embedding(u_cat_embed_idx)
        embed_out[:, self.preset_helper.matrix_categorical_bool_mask, :] = u_categorical_embeds
        # Numerical:
        u_numerical = u_in[:, self.preset_helper.matrix_numerical_bool_mask, 1:2]  # 3D tensor, not squeezed
        u_numerical_unmasked = self.numerical_embedding_conv(u_numerical.transpose(1, 2)).transpose(2, 1)
        u_numerical_embeds = u_numerical_unmasked[self.numerical_embedding_mask.expand(u_in.shape[0], -1, -1)]
        # This view is risky... but necessary because self.numerical_embedding_mask leads to a flattened tensor
        #   seems to work properly w/ pytorch 1.10, let's hope an update won't break the current behavior
        u_numerical_embeds = u_numerical_embeds.view(u_in.shape[0], self._n_num_params, self.hidden_size)
        embed_out[:, self.preset_helper.matrix_numerical_bool_mask, :] = u_numerical_embeds
        return embed_out

    # TODO should be usable for a single input param
    def forward_single_step(self, u_single_step: torch.Tensor):
        raise NotImplementedError()

