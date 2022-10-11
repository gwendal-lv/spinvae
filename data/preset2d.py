import copy
from typing import Optional

import numpy as np
import pandas as pd
import torch

from synth.dexed import Dexed
from .abstractbasedataset import PresetDataset


class Preset2dHelper:
    def __init__(self, ds: PresetDataset):
        """
        This class helps to convert a full VST preset (column vector of floats)
        into a matrix whose rows contain an 'extended' view of the preset, which can be used easily and
        properly by an Embedding layer.
        This class is to be used by Preset2d which actually performs conversions.

        Each matrix row contains three columns:
        (all values stored as float because of numerical values in [0.0, 1.0])
        - class (category) value
        - value of the discrete numerical value (-1 for categorical VST params)
        - VST parameter type (e.g. pitch, envelope, ...)

                       | categorical value | numerical value | param type
                       __________________________________________________
         param #0      |       -1.0             0.45              3.0
         param #1      |       4.0              -1.0              1.0
         param #2      |       -1.0             0.27              3.0
         param #....   |

        The class also provides masks to perform conversions quicker, to separate num/cat tensor in hidden
        sequence tensors, ...
        """
        self.synth_name = ds.synth_name
        self.torch_dtype = torch.float
        self._vst_param_learnable_model = ds.vst_param_learnable_model
        self._vst_idx_to_matrix_row = list()
        # np array to retrieve all vst values of a kind (num or cat) at once
        self.vst_numerical_bool_mask = np.zeros((ds.total_nb_vst_params, ), dtype=np.bool)
        self.vst_categorical_bool_mask = np.zeros((ds.total_nb_vst_params, ), dtype=np.bool)
        self.vst_params_card = np.asarray(
            [ds.get_preset_param_cardinality(vst_idx) for vst_idx in range(ds.total_nb_vst_params)], dtype=np.int)
        # Internal arrays of indices to access the different arrays quickly
        self.fixed_vst_indices = list()
        self._matrix_row_to_vst_idx = list()
        self.matrix_numerical_rows = list()
        self.matrix_numerical_rows_card = list()
        self.matrix_categorical_rows = list()
        self.matrix_categorical_rows_card = list()
        for vst_idx, learnable_model in enumerate(self._vst_param_learnable_model):
            # For conversions between a raw VST preset and its 2D tensor representation
            if learnable_model is not None:
                cur_matrix_row = len(self._matrix_row_to_vst_idx)  # Row that is going to be created
                self._vst_idx_to_matrix_row.append(cur_matrix_row)
                self._matrix_row_to_vst_idx.append(vst_idx)
                if learnable_model == 'num':
                    self.vst_numerical_bool_mask[vst_idx] = True
                    self.matrix_numerical_rows.append(cur_matrix_row)
                    self.matrix_numerical_rows_card.append(ds.get_preset_param_cardinality(vst_idx))
                elif learnable_model == 'cat':
                    self.vst_categorical_bool_mask[vst_idx] = True
                    self.matrix_categorical_rows.append(cur_matrix_row)
                    self.matrix_categorical_rows_card.append(ds.get_preset_param_cardinality(vst_idx))
                else:
                    raise ValueError("Unexpected _vst_param_learnable_model value '{}'".format(learnable_model))
            else:
                self._vst_idx_to_matrix_row.append(None)
                self.fixed_vst_indices.append(vst_idx)
        # Default VST param values: precompute a bool mask array and store the default values
        self.fixed_vst_params_bool_mask = np.zeros((ds.total_nb_vst_params, ), dtype=np.bool)
        self.fixed_vst_params_default_values = -1.0 * np.ones((len(self.fixed_vst_indices, )), dtype=np.float)
        for i, vst_idx in enumerate(self.fixed_vst_indices):
            self.fixed_vst_params_bool_mask[vst_idx] = True
            self.fixed_vst_params_default_values[i] = ds.params_default_values[vst_idx]
        self.max_cat_classes = max(self.matrix_categorical_rows_card)
        # Build tensor with the type of each learnable parameter (we don't care about fixed params set to default)
        self._param_types_str = [t for vst_idx, t in enumerate(ds.preset_param_types)
                                 if self._vst_param_learnable_model[vst_idx] is not None]
        self._unique_param_types_str = list(dict.fromkeys(self._param_types_str))  # dict is insertion-ordered (Python 3.7+)
        unique_types_indexes = {t: i for i, t in enumerate(self._unique_param_types_str)}
        self.param_types_tensor = torch.tensor(
            [unique_types_indexes[t] for t in self._param_types_str], dtype=self.torch_dtype
        )
        # Matrix (learnable tensor) Masks for categorical and numerical parameters
        self.matrix_numerical_bool_mask = torch.tensor(
            [i in self.matrix_numerical_rows for i in range(self.n_learnable_params)], dtype=torch.bool
        )
        self.matrix_numerical_mask = self.matrix_numerical_bool_mask.float()
        self.matrix_categorical_bool_mask = torch.tensor(
            [i in self.matrix_categorical_rows for i in range(self.n_learnable_params)], dtype=torch.bool
        )
        self.matrix_categorical_mask = self.matrix_categorical_bool_mask.float()
        # Build a "pre-filled" learnable preset tensor.
        self._pre_filled_matrix = -1 * torch.ones((self.n_learnable_params, 3), dtype=self.torch_dtype)
        self._pre_filled_matrix[:, 2] = self.param_types_tensor  # Set param types (will never change)
        # 2D Masks, that can be applied directly to the pre-filled matrix.
        # Pre-computed to be re-used quickly by a Preset2d instance
        # N.B. : the 1d masks will be useful for non-preset masking (e.g. for hidden sequence representations)
        self.matrix_numerical_bool_mask_2d = torch.unsqueeze(self.matrix_numerical_bool_mask, dim=1).repeat(1, 3)
        self.matrix_numerical_bool_mask_2d[:, 0] = False
        self.matrix_numerical_bool_mask_2d[:, 2] = False
        self.matrix_categorical_bool_mask_2d = torch.unsqueeze(self.matrix_categorical_bool_mask, dim=1).repeat(1, 3)
        self.matrix_categorical_bool_mask_2d[:, 1:3] = False

        # Categorical groups (1 group / cardinal)
        # Mask that must be applied to the reduced categorical-only sub-matrix
        self.categorical_groups_submatrix_bool_masks = dict()
        categorical_cardinals_set = sorted(list(set(self.matrix_categorical_rows_card)))
        for card in categorical_cardinals_set:
            self.categorical_groups_submatrix_bool_masks[card] = \
                np.asarray(self.matrix_categorical_rows_card) == card
        # Number of samples for each class, for each categorical group (NOT each single categorical synth param)
        self.categorical_groups_class_samples_counts = {
            card: np.zeros((card, ), dtype=np.int) for card in categorical_cardinals_set
        }
        for vst_idx, class_samples_count in ds.cat_params_class_samples_count.items():
            assert self.vst_categorical_bool_mask[vst_idx]
            card = class_samples_count.shape[0]
            self.categorical_groups_class_samples_counts[card] += class_samples_count

        # Retrieve params names (for plots) - numerical/categorical names can be extract from this one
        self.vst_params_names = np.array(ds.preset_param_names, copy=True)
        # Used in various classes: is type numerical, and retrieve the cardinality of a type of synth param
        self.is_type_numerical = [False] * self.n_param_types
        for matrix_row, param_type in enumerate(self.param_types_tensor.numpy()):
            self.is_type_numerical[int(param_type)] = self.matrix_numerical_bool_mask[matrix_row].item()
        self.param_type_to_cardinality = [None] * self.n_param_types
        # type2card: search for all params, and assert that all params of the same type have the same cardinality
        for matrix_idx, param_type in enumerate(self.param_types_tensor):
            type_class = int(param_type.item())
            vst_idx = self._matrix_row_to_vst_idx[matrix_idx]
            card = self.vst_params_card[vst_idx]
            if self.param_type_to_cardinality[type_class] is None:
                self.param_type_to_cardinality[type_class] = card
            else:
                assert self.param_type_to_cardinality[type_class] == card

        # Dexed synth-specific: bool masks to retrieve numerical/categorical rows that correspond to each operator
        if self.synth_name.lower() == "dexed":
            self.dexed_operators_categorical_bool_masks = \
                [torch.zeros((self.n_learnable_categorical_params, ), dtype=torch.bool) for _ in range(6)]
            self.dexed_operators_numerical_bool_masks = \
                [torch.zeros((self.n_learnable_numerical_params, ), dtype=torch.bool) for _ in range(6)]
            operators_vst_indices_groups = Dexed.get_operators_params_indexes_groups()
            operators_matrix_rows_groups = \
                [self._vst_idx_to_matrix_row[vst_range.start:vst_range.stop]
                 for vst_range in operators_vst_indices_groups]
            # Sub-optimal: for each idx of each group, search if it is numerical or categorical
            #   and find its "local row" inside the numerical or categorical split matrix
            for op_i, matrix_rows_group in enumerate(operators_matrix_rows_groups):
                for matrix_row in matrix_rows_group:
                    try:
                        row = self.matrix_numerical_rows.index(matrix_row)
                        self.dexed_operators_numerical_bool_masks[op_i][row] = True
                    except ValueError:
                        pass
                    try:
                        row = self.matrix_categorical_rows.index(matrix_row)
                        self.dexed_operators_categorical_bool_masks[op_i][row] = True
                    except ValueError:
                        pass
        else:
            self.dexed_operators_categorical_bool_masks, self.dexed_operators_numerical_bool_masks = None, None

    @property
    def n_learnable_params(self):
        """ Returns the number of learnable preset parameters i.e. the number of rows of the matrix (2D tensor)
        that is going to be fed to any encoder network. """
        return self.n_learnable_numerical_params + self.n_learnable_categorical_params

    @property
    def n_learnable_numerical_params(self):
        return len(self.matrix_numerical_rows)

    @property
    def n_learnable_categorical_params(self):
        return len(self.matrix_categorical_rows)

    @property
    def n_param_types(self):
        """ Return the number of different types of synth parameters. """
        return len(self._unique_param_types_str)

    @property
    def pre_filled_matrix(self):
        """ Returns a clone of the pre-filled matrix (learnable preset). """
        return self._pre_filled_matrix.clone()

    def get_null_learnable_preset(self, batch_size: Optional[int] = None):
        """
        Returns a null learnable preset (useful to build an output, to retrieve sizes, for debugging, ...).

        :param batch_size: If None, a 2D matrix will be returned. If int value is given, a batched 3D tensor
            will be returned.
        """
        t = self.pre_filled_matrix
        t[:, 0:2] = 0.0
        return t if batch_size is None else t.unsqueeze(0).repeat(batch_size, 1, 1)  # Repeat copies data

    @property
    def matrix_numerical_params_names(self):
        return self.vst_params_names[self.vst_numerical_bool_mask]

    @property
    def matrix_categorical_params_names(self):
        return self.vst_params_names[self.vst_categorical_bool_mask]

    #@property

    @property
    def pd_df_learnable_preset_debug(self):
        df = pd.DataFrame()
        df['preset_param_type_str'] = self._param_types_str
        df['preset_param_type'] = self.param_types_tensor.numpy()
        names = np.asarray(['' for _ in range(self.n_learnable_params)], dtype=self.vst_params_names.dtype)
        names[self.matrix_categorical_bool_mask] = self.matrix_categorical_params_names
        df['cat_name'] = names
        card_array = -10 * np.ones((self.n_learnable_params, ), dtype=np.int)
        for cat_idx, card in enumerate(self.matrix_categorical_rows_card):
            card_array[self.matrix_categorical_rows[cat_idx]] = card
        df['cat_card'] = card_array
        df['cat_bool_mask'] = self.matrix_categorical_bool_mask.numpy()
        df['cat_mask'] = self.matrix_categorical_mask.numpy()
        names = np.asarray(['' for _ in range(self.n_learnable_params)], dtype=self.vst_params_names.dtype)
        names[self.matrix_numerical_bool_mask] = self.matrix_numerical_params_names
        df['num_name'] = names
        card_array = -10 * np.ones((self.n_learnable_params, ), dtype=np.int)
        for num_idx, card in enumerate(self.matrix_numerical_rows_card):
            card_array[self.matrix_numerical_rows[num_idx]] = card
        df['num_card'] = card_array
        df['num_bool_mask'] = self.matrix_numerical_bool_mask.numpy()
        df['num_mask'] = self.matrix_numerical_mask.numpy()
        df['matrix_col_0 [cat]'] = self._pre_filled_matrix[:, 0]
        df['matrix_col_1 [num]'] = self._pre_filled_matrix[:, 1]
        df['matrix_col_2 [type]'] = self._pre_filled_matrix[:, 2]
        return df


class Preset2d:
    def __init__(self, ds: PresetDataset,
                 raw_vst_preset: Optional[np.ndarray] = None,
                 learnable_tensor_preset: Optional[torch.Tensor] = None,
                 force_fixed_params_default_values=True):
        """
        Class that can store a preset which comes from raw values (VST-ready floats)
         or from a learnable tensor (that could be fed to a neural network, or comes from the output of one).

        This class is the "new version" of PresetParams from preset.py and breaks compatibility.
        E.g., it works with non-batched presets
        (only 1 preset is held by this class, no unsqueezed singleton batch dim).

        :param raw_vst_preset:  1D numpy array which contains VST-compatible values
        :param learnable_tensor_preset:  2D tensor (matrix) with 2 or 3 columns. Column 0 must hold
            categorical data (class indices), column 1 must hold float numerical values. Column 2 should
            contain 'VST parameter type' class data, but is optional.
        :param force_fixed_params_default_values: For perfect consistency, default values should be
            written to any incoming raw preset
         """
        self.ds = ds
        self.preset_helper = ds.preset_indexes_helper  # type: Preset2dHelper
        assert isinstance(self.preset_helper, Preset2dHelper)
        assert raw_vst_preset is not None or learnable_tensor_preset is not None
        self._raw_preset, self._tensor_preset = raw_vst_preset, learnable_tensor_preset
        self._is_from_raw_preset = raw_vst_preset is not None
        if self.is_from_raw_preset:
            assert len(self._raw_preset.shape) == 1  # A single preset must be provided
            assert self._raw_preset.shape[0] == ds.total_nb_vst_params
            if force_fixed_params_default_values:
                self._raw_preset[self.preset_helper.fixed_vst_params_bool_mask] \
                    = self.preset_helper.fixed_vst_params_default_values
        else:
            assert len(self._tensor_preset.shape) == 2  # 2D tensors (unbatched presets)
            assert self._tensor_preset.shape[0] == self.preset_helper.n_learnable_params
            assert 2 <= self._tensor_preset.shape[1] <= 3
            # add 'type' column if necessary - otherwise the masks won't be usable
            if self._tensor_preset.shape[1] == 2:
                t = self.preset_helper.pre_filled_matrix
                t[:, 0:2] = self._tensor_preset.detach().clone().to(device='cpu')
                self._tensor_preset = t
            else:  # Move to CPU anyway
                self._tensor_preset = self._tensor_preset.detach().clone().to(device='cpu')

    @property
    def is_from_raw_preset(self):
        return self._is_from_raw_preset

    @property
    def is_from_learnable_preset(self):
        return not self._is_from_raw_preset

    def to_raw(self):
        if self.is_from_raw_preset:
            return self._raw_preset
        else:
            t = self._tensor_preset
            r = -1.0 * np.ones((self.ds.total_nb_vst_params, ))
            # retrieve the proper float value corresponding to categories,
            cat_n_classes = self.preset_helper.vst_params_card[self.preset_helper.vst_categorical_bool_mask]
            r[self.preset_helper.vst_categorical_bool_mask] \
                = t[self.preset_helper.matrix_categorical_bool_mask_2d] / (cat_n_classes - 1.0)
            # Then numerical value have to be copied
            r[self.preset_helper.vst_numerical_bool_mask] = t[self.preset_helper.matrix_numerical_bool_mask_2d]
            # Set default values (use pre-computed mask and values)
            r[self.preset_helper.fixed_vst_params_bool_mask] = self.preset_helper.fixed_vst_params_default_values
            return r

    def to_learnable_tensor(self):
        if self.is_from_learnable_preset:
            return self._tensor_preset
        else:
            t = self.preset_helper.pre_filled_matrix  # That property returns a clone
            # Column 0: classes (categorical synth params)
            cat_values = self._raw_preset[self.preset_helper.vst_categorical_bool_mask]
            cat_n_classes = self.preset_helper.vst_params_card[self.preset_helper.vst_categorical_bool_mask]
            cat_values *= (cat_n_classes - 1.0)
            t[self.preset_helper.matrix_categorical_bool_mask_2d] = torch.round(torch.tensor(cat_values))
            # Column 1: numerical values - easy: only need to copy using masks
            t[self.preset_helper.matrix_numerical_bool_mask_2d] \
                = torch.tensor(self._raw_preset[self.preset_helper.vst_numerical_bool_mask])
            return t

