from typing import List, Sequence

import numpy as np
import torch
import torchinfo
import warnings
from torch import nn as nn
from torch.nn import functional as F

from data.preset2d import Preset2dHelper
from model.presetmodel import parse_preset_model_architecture, get_act, PresetEmbedding
from utils.probability import GaussianUnitVariance



class PresetDecoder(nn.Module):
    def __init__(self, architecture: str,
                 latent_tensors_shapes: List[Sequence[int]],
                 hidden_size: int,
                 numerical_proba_distribution: str,
                 preset_helper: Preset2dHelper,
                 embedding: PresetEmbedding,
                 dropout_p=0.0, label_smoothing=0.0, use_cross_entropy_weights=False):
        """
        TODO DOC
        """
        super().__init__()

        # TODO add teacher-forcing input arg (and raise warning if non-sequential network)

        self.arch = parse_preset_model_architecture(architecture)
        arch_args = self.arch['args']
        self._latent_tensors_shapes = latent_tensors_shapes
        assert all([len(s) == 4 for s in self._latent_tensors_shapes]), "4d shapes (including batch size) are expected"
        self.dim_z = sum([np.prod(s[1:]) for s in self._latent_tensors_shapes])
        self.hidden_size = hidden_size
        self.preset_helper = preset_helper
        self.embedding = embedding
        self.seq_len = self.preset_helper.n_learnable_params

        # Warning: PyTorch broadcasting:
        #   - tensor broadcasting for (basic?) tensor operations requires the *trailing* dimensions to be equal
        #   - tensor default "broadcasting" for masks uses the mask on the 1st dimensions of the tensor to be masked
        # Masks don't need to be on the same device as the tensor they are masking
        self.seq_numerical_items_bool_mask = self.preset_helper.matrix_numerical_bool_mask.clone()
        self.seq_categorical_items_bool_mask = self.preset_helper.matrix_categorical_bool_mask.clone()

        # B) Categorical/Numerical final modules and probability distributions (required for A))
        # In hidden layers, the transformer does: "position-wise FC feed-forward network", equivalent to
        #     two 1-kernel convolution with ReLU in-between.
        # Output transformations are linear (probably position-wise...) then softmax to get the next-token probs;
        #     equivalent to conv1d, kernel 1, without bias
        self.categorical_module = MultiCardinalCategories(
            self.hidden_size, self.preset_helper,
            dropout_p=dropout_p, label_smoothing=label_smoothing, use_cross_entropy_weights=use_cross_entropy_weights
        )
        if numerical_proba_distribution == "gaussian_unitvariance":
            self.numerical_distrib = GaussianUnitVariance(
                mu_activation=nn.Hardtanh(0.0, 1.0), reduction='none'
            )
        else:
            raise NotImplementedError()
        self.numerical_distrib_conv1d = nn.Conv1d(
            self.hidden_size, self.numerical_distrib.num_parameters, 1, bias=False
        )

        # A) Build the main network (uses some self. attributes)
        if self.arch['name'] == 'mlp':
            # Feed-forward decoder
            # MLP is mostly used as a baseline/debugging model - so it uses its own quite big 2048 hidden dim
            self.child_decoder = MlpDecoder(self, mlp_hidden_features=2048)
        else:
            self.child_decoder = None
            # TODO Constraints on latent feature maps: size checks depend on the architecture (some are more flexible)
            #    currently support only single-level latents
            raise NotImplementedError("Preset architecture {} not implemented".format(self.arch['name']))

    def _move_masks_to(self, device):
        if self.seq_numerical_items_bool_mask.device != device:
            self.seq_numerical_items_bool_mask = self.seq_numerical_items_bool_mask.to(device)
            self.seq_categorical_items_bool_mask = self.seq_categorical_items_bool_mask.to(device)

    def forward(self, z_multi_level: List[torch.Tensor], u_target: torch.Tensor):
        """

        :param z_multi_level: Deepest latent values are the last elements of this list
        :param u_target: Input preset, sequence-like (expected shape: N x n_synth_presets x 3)
        :return:
        """
        self._move_masks_to(u_target.device)

        # ----- A) Apply the "main" feed-forward or recurrent / sequential network -----$
        u_out, u_categorical_nll, u_categorical_samples, u_numerical_nll, u_numerical_samples \
            = self.child_decoder(z_multi_level, u_target)

        # ----- B) Compute metrics (quite easy, we can do it now) -----
        # we don't return separate metric for teacher-forcing
        #    -> because teacher-forcing will be activated or not from the owner of this module
        with torch.no_grad():
            num_l1_error = torch.mean(
                torch.abs(u_target[:, self.seq_numerical_items_bool_mask, 1] - u_numerical_samples), dim=1
            )
            acc = torch.eq(u_target[:, self.seq_categorical_items_bool_mask, 0].long(), u_categorical_samples)
            acc = acc.count_nonzero(dim=1) / acc.shape[1]

        # sum NLLs and divide by total sequence length (num of params) - keep batch dimension (if multi-GPU)
        u_numerical_nll = u_numerical_nll.sum(dim=1) / self.seq_len
        u_categorical_nll = u_categorical_nll.sum(dim=1) / self.seq_len
        return u_out, \
            u_numerical_nll, u_categorical_nll, \
            num_l1_error, acc

    def get_summary(self, device='cpu'):
        device = torch.device(device)
        input_z = [torch.rand(s).to(device) - 0.5 for s in self._latent_tensors_shapes]
        input_u = self.preset_helper.get_null_learnable_preset(self._latent_tensors_shapes[0][0]).to(device)
        with torch.no_grad():
            return torchinfo.summary(
                self, input_data=(input_z, input_u),
                depth=5, verbose=0, device=device,
                col_names=("input_size", "kernel_size", "output_size", "num_params", "mult_adds"),
                row_settings=("depth", "var_names")
            )


class ChildDecoderBase(nn.Module):
    def __init__(self, parent_dec: PresetDecoder):
        """
        Base class for any "sub-decoder" child (e.g. Mlp, RNN, Transformer) of a PresetDecoder instance.
        This allows to easily share useful instances and information with all children.
        """
        super().__init__()
        self.seq_len = parent_dec.seq_len
        self.dim_z = parent_dec.dim_z
        self.hidden_size = parent_dec.hidden_size
        self.preset_helper = parent_dec.preset_helper
        self.embedding = parent_dec.embedding
        self.arch_args = parent_dec.arch['args']
        self.n_layers = parent_dec.arch['n_layers']

        self.categorical_module = parent_dec.categorical_module
        self.numerical_distrib_conv1d = parent_dec.numerical_distrib_conv1d
        self.numerical_distrib = parent_dec.numerical_distrib
        self.seq_numerical_items_bool_mask = parent_dec.seq_numerical_items_bool_mask
        self.seq_categorical_items_bool_mask = parent_dec.seq_categorical_items_bool_mask


class MlpDecoder(ChildDecoderBase):
    def __init__(self, parent_dec: PresetDecoder,mlp_hidden_features=2048, sequence_dim_last=True):
        """
        Simple module that reshapes the output from an MLP network into a 2D feature map with seq_len as
        its first dimension (number of channels will be inferred automatically), and applies a 1x1 conv
        to increase the number of channels to seq_hidden_dim.

        :param sequence_dim_last: If True, the sequence dimension will be the last dimension (use this when
            the output of this module is used by a CNN). If False, the features (channels) dimension will be last
            (for use as RNN input).
        """
        super().__init__(parent_dec)
        # self.hidden_size is NOT the number of hidden neurons in this MLP (it's the feature vector size, for
        #  recurrent networks only - not applicable to this MLP).
        self.seq_hidden_dim = self.hidden_size

        self.mlp = nn.Sequential()
        n_pre_out_features = self.seq_len * (mlp_hidden_features // self.seq_len)
        for l in range(0, self.n_layers):
            if l > 0:
                if l < self.n_layers - 1:  # No norm before the last layer
                    if self.arch_args['bn']:
                        self.mlp.add_module('bn{}'.format(l), nn.BatchNorm1d(mlp_hidden_features))
                    if self.arch_args['ln']:
                        raise NotImplementedError()
                self.mlp.add_module('act{}'.format(l), get_act(self.arch_args))
            n_in_features = mlp_hidden_features if (l > 0) else self.dim_z
            n_out_features = mlp_hidden_features if (l < self.n_layers - 1) else n_pre_out_features
            self.mlp.add_module('fc{}'.format(l), nn.Linear(n_in_features, n_out_features))

        if self.arch_args['posenc']:
            warnings.warn("Cannot use positional embeddings with an MLP network - ignored")

        # Last layer should output a 1d tensor that can be reshaped as a "sequence-like" 2D tensor.
        #    This would represent an overly huge FC layer.... so it's done in 2 steps (See ReshapeMlpOutput)
        self.in_channels = n_pre_out_features // self.seq_len
        self.conv = nn.Sequential(
            get_act(self.arch_args),
            nn.Conv1d(self.in_channels, self.seq_hidden_dim, 1)
        )
        self.sequence_dim_last = sequence_dim_last

    def forward(self, z_multi_level, u_target):
        # ---------- 1) Apply the feed-forward MLP ----------
        z_flat = torch.cat([z.flatten(start_dim=1) for z in z_multi_level], dim=1)
        u_hidden = self.mlp(z_flat).view(-1, self.in_channels, self.seq_len)
        u_hidden = self.conv(u_hidden)  # After this conv, sequence dim ("time" or "step" dimension) is last

        u_out = self.preset_helper.get_null_learnable_preset(u_target.shape[0]).to(u_target.device)
        # ---------- 2) Compute parameters of probability distributions, compute NLLs (all at once) ----------
        # At this point, u_last_hidden is the hidden sequence representation of the preset
        #     shape must be: hidden_size x L (sequence dimension last)
        #
        # --- Categorical distribution(s) ---
        u_categorical_logits, u_categorical_nll, u_categorical_samples = self.categorical_module.forward_full_sequence(
            u_hidden[:, :, self.seq_categorical_items_bool_mask],
            u_target[:, self.seq_categorical_items_bool_mask, 0]
        )
        u_out[:, self.seq_categorical_items_bool_mask, 0] = u_categorical_samples.float()
        # --- Numerical distribution (s) ---
        # FIXME different distributions for parameters w/ a different cardinal
        numerical_distrib_params = self.numerical_distrib_conv1d(
            u_hidden[:, :, self.seq_numerical_items_bool_mask]
        )
        numerical_distrib_params = self.numerical_distrib.apply_activations(numerical_distrib_params)
        # These are NLLs and samples for a "single-channel" distribution -> squeeze for consistency vs. categorical
        u_numerical_nll = torch.squeeze(self.numerical_distrib.NLL(
            numerical_distrib_params,
            u_target[:, self.seq_numerical_items_bool_mask, 1:2].transpose(2, 1)  # Set sequence dim last
        ))
        u_numerical_samples = torch.squeeze(self.numerical_distrib.sample(numerical_distrib_params))
        u_out[:, self.seq_numerical_items_bool_mask, 1] = u_numerical_samples

        return u_out, u_categorical_nll, u_categorical_samples, u_numerical_nll, u_numerical_samples


class MultiCardinalCategories(nn.Module):
    def __init__(self, hidden_size: int, preset_helper: Preset2dHelper,
                 dropout_p=0.0, label_smoothing=0.0, use_cross_entropy_weights=False):
        """
        This class is able to compute the categorical logits using several groups, and to compute the CE loss
        for each synth param of each group.
        Each group contains categorical synth params with the same number of classes (cardinal of the set of values).
        Each group has a different number of logits (using only 1 group with masking could lead to a biased
         preference towards the first logits).
        """
        super().__init__()
        self.preset_helper = preset_helper
        self.label_smoothing = label_smoothing
        self.use_ce_weights = use_cross_entropy_weights
        # 1 conv1d for each categorical group - 4 groups instead of 1 (the overhead should be acceptable...)
        self.categorical_distribs_conv1d = dict()
        self.cross_entropy_weights = dict()
        for card, mask in self.preset_helper.categorical_groups_submatrix_bool_masks.items():
            self.categorical_distribs_conv1d[card] = nn.Sequential(
                nn.Dropout(dropout_p),
                nn.Conv1d(hidden_size, card, 1, bias=False)
            )
            # Compute some "merged class counts" or "weights". We don't instantiate a CE loss because we don't know
            # the device yet (and Pytorch's CE class is a basic wrapper that calls F.cross_entropy)
            class_counts = self.preset_helper.categorical_groups_class_samples_counts[card] + 10  # 10 is an "epsilon"
            weights = 1.0 - (class_counts / class_counts.sum())
            # average weight will be 1.0
            self.cross_entropy_weights[card] = torch.tensor(weights / weights.mean(), dtype=torch.float)

        # This dict allows nn parameters to be registered by PyTorch
        self._mods_dict = nn.ModuleDict(
            {'card{}'.format(card): m for card, m in self.categorical_distribs_conv1d.items()}
        )
        # Constant sizes
        self.Lc = self.preset_helper.n_learnable_categorical_params

    def forward(self):
        raise NotImplementedError("Use forward_full_sequence or forward_item.")

    def forward_full_sequence(self, u_categorical_last_hidden, u_target: torch.Tensor):
        """

        :param u_categorical_last_hidden: Tensor of shape N x hidden_size x Lc
            where Lc is the total number of categorical output variables (synth params)
        :param u_target: target tensor, shape N x Lc
        :return: out_logits (dict, different shapes),
                    out_ce (shape N x Lc) (is a NLL),
                    samples_categories (shape N x Lc)
        """
        u_target = u_target.long()
        out_logits = dict()
        out_ce = torch.empty((u_categorical_last_hidden.shape[0], self.Lc), device=u_categorical_last_hidden.device)
        sampled_categories = torch.empty(
            (u_categorical_last_hidden.shape[0], self.Lc), dtype=torch.long, device=u_categorical_last_hidden.device)
        for card, module in self.categorical_distribs_conv1d.items():
            mask = self.preset_helper.categorical_groups_submatrix_bool_masks[card]
            logits = module(u_categorical_last_hidden[:, :, mask])
            # logits is the output of a conv-like module, so the sequence dim is dim=2, and
            # class probabilities are dim=1 (which is expected when calling F.cross_entropy
            sampled_categories[:, mask] = torch.argmax(logits, dim=1, keepdim=False)
            # Don't turn off label smoothing during validation (we don't want validation over-confidence either)
            # However, we don't use the (training) weights during validation
            if self.use_ce_weights and self.training:
                weight = self.cross_entropy_weights[card].to(u_target.device)
            else:
                weight = None
            out_ce[:, mask] = F.cross_entropy(
                logits, u_target[:, mask],
                reduction='none', label_smoothing=self.label_smoothing, weight=weight
            )
            out_logits[card] = logits
        return out_logits, out_ce, sampled_categories

    def forward_item(self, u_categorical_last_hidden, categorical_submatrix_index: int):
        """

        :param u_categorical_last_hidden: Tensor of shape TODO SHAPE
        :param categorical_submatrix_index: Sequence-index of the current sequence item (index in [0, Lc - 1])
        :return:
        """
        card_group = self.preset_helper.matrix_categorical_rows_card[categorical_submatrix_index]
        raise NotImplementedError()  # FIXME this has to be checked before being used
