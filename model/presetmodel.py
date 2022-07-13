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

        # is type numerical - to be able to retrieve the embedding inside forward_single_step(...)
        self.is_type_numerical = self.preset_helper.is_type_numerical

        # TODO ctor arg to use "naive basic" transforms, identical for all synth params (interesting experiment)

        # Numerical: use a 1d conv with a lot of output channels
        n_numerical_conv_ch = hidden_size * preset_helper.n_param_types
        # FIXME this method uses very similar embeddings for different discrete numerical synth params
        # TODO try add a learned bias for each param type?
        #   Or maybe an additionnal global embedding for each param type... (seen in IJCAI19 "T-CVAE story completion")
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
        # FIXME don't use so much unnecessary embeddings
        n_categorical_embeddings = preset_helper.max_cat_classes * preset_helper.n_param_types
        self.categorical_embedding = nn.Embedding(n_categorical_embeddings, hidden_size, max_norm=None)

        # Save other masks - they are will be moved to the GPU
        self._matrix_categorical_bool_mask = self.preset_helper.matrix_categorical_bool_mask.clone()
        self._matrix_numerical_bool_mask = self.preset_helper.matrix_numerical_bool_mask.clone()

        # Precompute the largest possible embedding: seq w/ start and end tokens
        self.pos_embed_L_plus_2 = self.get_sin_cos_positional_embedding(seq_len=self.seq_len + 2)

        # TODO unit test: assert that all embeddings are different (after random init)

    @property
    def seq_len(self):
        return self.preset_helper.n_learnable_params

    @property
    def _n_num_params(self):
        return self.preset_helper.n_learnable_numerical_params

    def _get_categorical_embed_idx(self, u_categorical_only: torch.Tensor):
        return torch.round(
            u_categorical_only[:, :, 2] * self.preset_helper.max_cat_classes + u_categorical_only[:, :, 0]
        ).long()

    def _move_masks_and_embeds_to(self, device):
        if self.numerical_embedding_mask.device != device:
            # numerical_embedding_mask is huge so this leads to a significantly reduced epoch duration
            self.numerical_embedding_mask = self.numerical_embedding_mask.to(device)
            self._matrix_numerical_bool_mask = self._matrix_numerical_bool_mask.to(device)
            self._matrix_categorical_bool_mask = self._matrix_categorical_bool_mask.to(device)
            self.pos_embed_L_plus_2 = self.pos_embed_L_plus_2.to(device)

    def get_start_token(self, device, batch_dim=False):
        if batch_dim:
            return torch.zeros((1, 1, self.hidden_size), device=device)
        else:
            return torch.zeros((1, self.hidden_size), device=device)

    def forward(self, u_in: torch.Tensor, start_token=False, pos_embed=True):
        """
        Returns an embedding, shape N x L x hidden_size if start_token is False else N x (L+1) x hidden_size
        corresponding to an entire input preset (shape N x L x 3)

        :param start_token: If True, a special start token will be inserted at the beginning of the sequence,
            but the last item will NOT be discarded.
        :param pos_embed: If True, Transformer positional embeddings will be added.
        """
        self._move_masks_and_embeds_to(u_in.device)
        N = u_in.shape[0]  # minibatch size
        embed_out = torch.empty((N, u_in.shape[1], self.hidden_size), device=u_in.device)
        # Categorical: Class values in "column" 0, types in "column" 2 ("column" is the last dimension)
        u_categorical = u_in[:, self._matrix_categorical_bool_mask, :]
        u_cat_embed_idx = self._get_categorical_embed_idx(u_categorical)
        u_categorical_embeds = self.categorical_embedding(u_cat_embed_idx)
        embed_out[:, self._matrix_categorical_bool_mask, :] = u_categorical_embeds
        # Numerical:
        u_numerical = u_in[:, self._matrix_numerical_bool_mask, 1:2]  # 3D tensor, not squeezed
        # FIXME URGENT !
        #     1) use linear instead of conv
        #     2) CENTER numerical values?
        u_numerical_unmasked = self.numerical_embedding_conv(u_numerical.transpose(1, 2)).transpose(2, 1)
        u_numerical_embeds = u_numerical_unmasked[self.numerical_embedding_mask.expand(N, -1, -1)]
        # This view is risky... but necessary because self.numerical_embedding_mask leads to a flattened tensor
        #   seems to work properly w/ pytorch 1.10, let's hope an update won't break the current behavior
        u_numerical_embeds = u_numerical_embeds.view(N, self._n_num_params, self.hidden_size)
        embed_out[:, self._matrix_numerical_bool_mask, :] = u_numerical_embeds
        # insert start token (we do not it at the end, not to mess with masks)
        if start_token:  # TODO learnable start token?
            start_embed = self.get_start_token(u_in.device, batch_dim=True).expand(N, -1, -1)
            embed_out = torch.cat((start_embed, embed_out), dim=1)  # Concat along seq dim, start token first
            if pos_embed:
                embed_out += self.pos_embed_L_plus_1  # broadcast over minibatch dimension
        else:
            if pos_embed:
                embed_out += self.pos_embed_L  # broadcast over minibatch dimension
        return embed_out

    def forward_single_token(self, u_single_step: torch.Tensor):
        """ Returns the embedding (shape N x 1 x hidden_size) of a single-step input element (shape N x 1 x 3).
        This method is NOT able to add any positional embedding - it has to be done outside of this method. """
        assert u_single_step.shape[1] == 1
        type_class = int(u_single_step[0, 0, 2].item())  # Types (~= positions) are the same for all items from a batch
        if self.is_type_numerical[type_class]:  # Numerical type
            embed = self.numerical_embedding_conv(u_single_step[:, :, 1:2].transpose(1, 2)).transpose(2, 1)
            # We can use the known range instead of using a mask
            return embed[:, :, (type_class * self.hidden_size): ((type_class + 1) * self.hidden_size)]
        else:  # Categorical type
            return self.categorical_embedding(self._get_categorical_embed_idx(u_single_step))

    def get_sin_cos_positional_embedding(self, max_len=10000.0, seq_len=None):
        D = self.hidden_size
        # Sequence length can be increased w/ start/stop tokens
        L = seq_len if seq_len is not None else self.seq_len
        embed = torch.unsqueeze(torch.arange(0, L, dtype=torch.int).float(), dim=1)
        embed = embed.repeat(1, D)
        for i in range(D//2):
            omega_inverse = max_len ** (2.0 * i / D)
            embed[:, 2 * i] = torch.sin(embed[:, 2 * i] / omega_inverse)
            embed[:, 2 * i + 1] = torch.cos(embed[:, 2 * i + 1] / omega_inverse)
        return embed

    @property
    def pos_embed_L(self):
        return self.pos_embed_L_plus_2[0:self.seq_len, :]

    @property
    def pos_embed_L_plus_1(self):
        return self.pos_embed_L_plus_2[0:self.seq_len + 1, :]

