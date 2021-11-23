"""
Neural networks classes for synth parameters regression, and related utility functions.

These regression models can be used on their own, or passed as constructor arguments to
extended AE models.
"""
import threading
from collections.abc import Iterable
from abc import ABC, abstractmethod  # Abstract Base Class

import torch
import torch.nn as nn
import torch.nn.functional as F

from nflows.transforms.base import CompositeTransform
from nflows.transforms.autoregressive import MaskedAffineAutoregressiveTransform
from nflows.transforms.permutations import ReversePermutation

from data.preset import PresetIndexesHelper
import model.base
import model.flows
from model.flows import CustomRealNVP, InverseFlow
import model.loss


class PresetActivation(nn.Module):
    """ Applies the appropriate activations (e.g. sigmoid, hardtanh, softmax, ...) to different neurons
    or groups of neurons of a given input layer. """
    def __init__(self, idx_helper: PresetIndexesHelper,
                 cat_hardtanh_activation=False, cat_softmax_activation=False):
        """
        Applies a [0;1] hardtanh activation on numerical parameters, and activates or not the categorical sub-vectors.

        :param idx_helper:
        :param activate_cat: if True, applies a [0;1] hardtanh activation to categorical outputs
        :param cat_softmax_activation: if True, a softmax activation is applied on categorical sub-vectors.
            Otherwise, applies the same HardTanh for cat and num params (and softmax should be applied in loss function)
        """
        super().__init__()
        self.idx_helper = idx_helper
        # Pre-compute indexes lists (to use less CPU)
        self.num_indexes = self.idx_helper.get_numerical_learnable_indexes()
        self.cat_indexes = self.idx_helper.get_categorical_learnable_indexes()  # type: Iterable[Iterable]
        self.cat_indexes_1d_list = sum(self.cat_indexes, [])

        self.hardtanh_act = nn.Hardtanh(min_val=0.0, max_val=1.0)
        self.cat_hardtanh_activation = cat_hardtanh_activation
        self.cat_softmax_activation = cat_softmax_activation
        if self.cat_softmax_activation:
            self.categorical_act = nn.Softmax(dim=-1)  # Required for categorical cross-entropy loss
        else:
            self.categorical_act = None

    def forward(self, x):
        """ Applies per-parameter output activations using the PresetIndexesHelper attribute of this instance. """
        x[:, self.num_indexes] = self.hardtanh_act(x[:, self.num_indexes])
        if self.cat_hardtanh_activation:
            x[:, self.cat_indexes_1d_list] = self.hardtanh_act(x[:, self.cat_indexes_1d_list])
        if self.cat_softmax_activation:
            for cat_learnable_indexes in self.cat_indexes:  # type: Iterable
                x[:, cat_learnable_indexes] = self.categorical_act(x[:, cat_learnable_indexes])
        return x



class RegressionModel(model.base.TrainableModel, ABC):
    def __init__(self, architecture: str, dim_z: int, idx_helper: PresetIndexesHelper,
                 cat_hardtanh_activation=False, cat_softmax_activation=False,
                 model_config=None, train_config=None):
        super().__init__(train_config=train_config, model_type='reg')
        self.architecture = architecture
        self.arch_args = architecture.split('_')  # Split between base args and opt args (e.g. _nobn)
        self.dim_z = dim_z
        self.idx_helper = idx_helper

        self.activation_layer = PresetActivation(
            self.idx_helper, cat_hardtanh_activation=cat_hardtanh_activation,
            cat_softmax_activation=cat_softmax_activation)

        # Attributes used for training only (losses, ...)
        if train_config is not None and model_config is not None:
            self.dropout_p = train_config.reg_fc_dropout
            self.loss_wih_permutations = train_config.params_loss_with_permutations
            if train_config.params_cat_bceloss and model_config.params_reg_softmax:
                raise AssertionError("BCE loss requires no-softmax at reg model output")
            self._backprop_criterion = model.loss.SynthParamsLoss(
                self.idx_helper, train_config.normalize_losses, cat_bce=train_config.params_cat_bceloss,
                cat_softmax=(not model_config.params_reg_softmax and not train_config.params_cat_bceloss),
                cat_softmax_t=train_config.params_cat_softmax_temperature,
                prevent_useless_params_loss=train_config.params_loss_exclude_useless,
                compute_symmetrical_presets=self.loss_wih_permutations
            )
            # Monitoring losses always remain the same (always return the loss corresponding to the best permutation)
            self._eval_criterion = model.loss.AccuracyAndQuantizedNumericalLoss(
                self.idx_helper, numerical_loss_type='L1', reduce_accuracy=True, percentage_accuracy_output=True,
                compute_symmetrical_presets=True
            )
        else:
            self._backprop_criterion, self._eval_criterion = None, None
            self.dropout_p = 0.0
        # Those will be stored during train or eval, after associated method have been called
        self.current_u_in, self.current_u_out = None, None
        self.current_permutation_groups, self.current_u_in_w_s, self.current_u_out_w_s = None, None, None
        self._permutations_computation_thread = None

    @abstractmethod
    def _reg_model_without_activation(self, z):
        pass

    def forward(self, z):
        """ Applies the regression model to a z latent vector (VAE latent flow output samples). """
        return self.activation_layer(self.reg_model(z))

    def precompute_u_in_permutations(self, u_in):
        """ Computes possible permutations (i.e. different presets that give the exact same sonic output, using
         symmetries of the synth) of the u_in presets, and stores them internally.
         Permutations are always used by evaluation criteria, so they must always be computed. """
        self.current_u_in, self.current_u_out = u_in, None
        self.current_permutation_groups, self.current_u_in_w_s, self.current_u_out_w_s = None, None, None
        self._permutations_computation_thread = threading.Thread(
            target=self._precompute_u_in_permutations__thread, args=(u_in, ))
        self._permutations_computation_thread.start()

    def _precompute_u_in_permutations__thread(self, u_in):
        self.current_permutation_groups, self.current_u_in_w_s = self.idx_helper.get_u_in_permutations(u_in)

    def precompute_u_out_with_symmetries(self, u_out):
        self._permutations_computation_thread.join()
        self._permutations_computation_thread = None
        # This is done directly on GPU - no need to use a parallel thread
        self.current_u_out = u_out
        if self.current_permutation_groups is None:
            raise AssertionError("self.current_permutation_groups have not been computed yet.")
        self.current_u_out_w_s = self.idx_helper.get_u_out_permutations(self.current_permutation_groups, u_out)

    def _assert_pre_computed_symmetries_available(self):
        if self.current_u_in is None or self.current_u_in_w_s is None or self.current_permutation_groups is None:
            raise AssertionError("The precompute_u_in_permutations(..) method must be called before this one.")
        if self.current_u_out is None or self.current_u_out_w_s is None:
            raise AssertionError("The precompute_u_out_with_symmetries(..) method must be called before this one.")

    @property
    def backprop_loss_value(self):
        """ Returns the current backprop loss value, computed using previously given u_in and u_out values
        (see precompute_u_in_permutations(...) and precompute_u_out_with_symmetries(...) methods. """
        if self.loss_wih_permutations:
            self._assert_pre_computed_symmetries_available()
            return self._backprop_criterion.loss_with_permutations(
                self.current_permutation_groups, self.current_u_out_w_s, self.current_u_in_w_s)
        else:  # no permutation: we let the loss do its job
            return self._backprop_criterion(self.current_u_out, self.current_u_in)

    @property
    def eval_criterion_values(self):
        """ Returns the current eval accuracy and numerical error, computed using previously given u_in and u_out values
        (see precompute_u_in_permutations(...) and precompute_u_out_with_symmetries(...) methods. """
        self._assert_pre_computed_symmetries_available()
        return self._eval_criterion.losses_with_permutations(
            self.current_permutation_groups, self.current_u_out_w_s, self.current_u_in_w_s)

    def find_preset_inverse(self, u_target: torch.Tensor, z_first_guess: torch.Tensor):
        if u_target.shape[0] != 1 or z_first_guess.shape[0] != 1:  # check 1d input (batch of 1 element)
            raise AssertionError("This method handles only 1 preset at a time.")
        z_start = torch.tensor(z_first_guess, requires_grad=False)  # Constant tensor
        z_offset = torch.zeros_like(z_start, requires_grad=True)  # We'll optimize this one
        optimizer = torch.optim.Adam([z_offset], lr=0.1, weight_decay=1e-5)
        backprop_criterion = model.loss.SynthParamsLoss(
            self.idx_helper, normalize_losses=True,
            compute_symmetrical_presets=False, prevent_useless_params_loss=False,
        )
        eval_criterion = model.loss.AccuracyAndQuantizedNumericalLoss(
            self.idx_helper, compute_symmetrical_presets=False
        )
        # Searching loop
        acc, num_loss = 0.0, 1.0
        for i in range(100):  # FIXME dummy test
            optimizer.zero_grad()
            z_estimated = z_start + z_offset
            u_out = self.forward(z_estimated)
            backprop_loss = backprop_criterion(u_out, u_target)
            backprop_loss.backward()
            optimizer.step()
            acc, num_loss = eval_criterion(u_out, u_target)
            if i%10 == 0:
                print("Accuracy = {:.1f}%, num_loss = {:.3f}".format(acc, num_loss))
        z_offset.requires_grad = False
        return z_start + z_offset, acc, num_loss



class MLPControlsRegression(RegressionModel):
    def __init__(self, architecture: str, dim_z: int, idx_helper: PresetIndexesHelper,
                 cat_hardtanh_activation=False, cat_softmax_activation=False,
                 model_config=None, train_config=None):
        """
        :param architecture: MLP automatically built from architecture string. E.g. '3l1024' means
            3 hidden layers of 1024 neurons. Some options can be given after an underscore
            (e.g. '3l1024_nobn' adds the no batch norm argument). See implementation for more details.  TODO implement
        :param dim_z: Size of a z_K latent vector
        :param idx_helper:
        """
        super().__init__(architecture, dim_z, idx_helper, cat_hardtanh_activation, cat_softmax_activation,
                         model_config, train_config)
        if len(self.arch_args) == 1:
            num_hidden_layers, num_hidden_neurons = self.arch_args[0].split('l')
            num_hidden_layers, num_hidden_neurons = int(num_hidden_layers), int(num_hidden_neurons)
        else:
            raise NotImplementedError("Arch suffix arguments not implemented yet")
        # Layers definition
        self.reg_model = nn.Sequential()
        for l in range(0, num_hidden_layers):
            if l == 0:
                self.reg_model.add_module('fc{}'.format(l + 1), nn.Linear(dim_z, num_hidden_neurons))
            else:
                self.reg_model.add_module('fc{}'.format(l + 1), nn.Linear(num_hidden_neurons, num_hidden_neurons))
            # No BN or dropouts in the 2 last FC layers
            # Dropout in the deepest hidden layers is absolutely necessary (strong overfit otherwise).
            if l < (num_hidden_layers - 1):
                self.reg_model.add_module('bn{}'.format(l + 1), nn.BatchNorm1d(num_features=num_hidden_neurons))
                self.reg_model.add_module('drp{}'.format(l + 1), nn.Dropout(self.dropout_p))
            self.reg_model.add_module('act{}'.format(l + 1), nn.ReLU())
        self.reg_model.add_module('fc{}'.format(num_hidden_layers + 1), nn.Linear(num_hidden_neurons,
                                                                                  self.idx_helper.learnable_preset_size))

    def _reg_model_without_activation(self, z):
        return self.reg_model(z)


class FlowControlsRegression(RegressionModel):
    def __init__(self, architecture: str, dim_z: int, idx_helper: PresetIndexesHelper,
                 fast_forward_flow=True,
                 cat_hardtanh_activation=False, cat_softmax_activation=False,
                 model_config=None, train_config=None):
        """
        :param architecture: Flow automatically built from architecture string. E.g. 'realnvp_16l200' means
            16 RealNVP flow layers with 200 hidden features each. Some options can be given after an underscore
            (e.g. '16l200_bn' adds batch norm). See implementation for more details.  TODO implement suffix options
        :param dim_z: Size of a z_K latent vector, which is also the output size for this invertible normalizing flow.
        :param idx_helper:
        :param dropout_p:
        :param fast_forward_flow: If True, the flow transform will be built such that it is fast (and memory-efficient)
            in the forward direction (else, it will be fast in the inverse direction). Moreover, if batch-norm is used
            between layers, the flow can be trained only its 'fast' direction (which can be forward or inverse
            depending on this argument).
        """
        super().__init__(architecture, dim_z, idx_helper, cat_hardtanh_activation, cat_softmax_activation,
                         model_config, train_config)
        self._fast_forward_flow = fast_forward_flow
        self.flow_type, self.num_flow_layers, self.num_flow_hidden_features, _, _, _ = \
            model.flows.parse_flow_args(architecture, authorize_options=False)
        # Default: BN usage everywhere but between the 2 last layers
        self.bn_between_flows = True
        self.bn_within_flows = True
        self.bn_output = False

        # Multi-layer flow definition
        if self.flow_type.lower() == 'realnvp' or self.flow_type.lower() == 'rnvp':
            # RealNVP - custom (without useless gaussian base distribution) and no BN on last layers
            self._forward_flow_transform = CustomRealNVP(
                self.dim_z, self.num_flow_hidden_features, self.num_flow_layers, dropout_probability=self.dropout_p,
                bn_between_layers=self.bn_between_flows, bn_within_layers=self.bn_within_flows,
                output_bn=self.bn_output)
        elif self.flow_type.lower() == 'maf':
            transforms = []
            for l in range(self.num_flow_layers):
                transforms.append(ReversePermutation(features=self.dim_z))
                # TODO Batch norm added on all flow MLPs but the 2 last
                #     and dropout p
                transforms.append(MaskedAffineAutoregressiveTransform(features=self.dim_z,
                                                                      hidden_features=self.num_flow_hidden_features,
                                                                      use_batch_norm=False,  # TODO (l < num_layers-2),
                                                                      dropout_probability=0.5  # TODO as param
                                                                      ))
            self._forward_flow_transform = CompositeTransform(transforms)  # Fast forward  # TODO rename
            # The inversed MAF flow should never (cannot...) be used during training:
            #   - much slower than forward (in nflows implementation)
            #   - very unstable
            #   - needs ** huge ** amounts of GPU RAM
        else:
            raise ValueError("Undefined flow type '{}'".format(self.flow_type))

    def _reg_model_without_activation(self, z):
        v_out, _ = self.flow_forward_function(z)
        return v_out

    @property
    def is_flow_fast_forward(self):  # TODO improve, real nvp is fast forward and inverse...
        return self._fast_forward_flow

    @property
    def flow_forward_function(self):
        if self._fast_forward_flow:
            return self._forward_flow_transform.forward
        else:
            return self._forward_flow_transform.inverse

    @property
    def flow_inverse_function(self):
        if not self._fast_forward_flow:
            return self._forward_flow_transform.forward
        else:
            return self._forward_flow_transform.inverse




