
from typing import Iterable, Sequence, Optional, List
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from nflows.transforms.base import CompositeTransform

from data.preset import PresetIndexesHelper
import utils.probability


class L2Loss:
    """
    L2 (squared difference) loss, with customizable normalization (averaging) options.

    When used to model the reconstruction probability p_theta( x | zK ), normalization has strong
    implications on the p_theta( x | zK ) model itself.
    E.g., for a 1-element batch, the non-normalized L2 loss implies a learned mean, fixed 1/√2 std
    gaussian model for each element of x.
    When normalizing the L2 error (i.e. MSE error), the fixed std is multiplied by √(nb of elements of x)
    (e.g. approx *300 for a 250x350 pixels spectrogram)

    Normalization over batch dimension should always be performed (monte-carlo log-proba estimation).
    """
    def __init__(self, contents_average=False, batch_average=True):
        """

        :param contents_average: If True, the loss value will be divided by the number of elements of a batch item.
        :param batch_average: If True, the loss value will be divided by batch size
        """
        self.contents_average = contents_average
        self.batch_average = batch_average

    def __call__(self, inferred, target):
        loss = torch.sum(torch.square(inferred - target))
        if self.batch_average:
            loss = loss / inferred.shape[0]
        if self.contents_average:
            loss = loss / inferred[0, :].numel()
        return loss


class GaussianDkl:
    """ Kullback-Leibler Divergence between independant Gaussian distributions (diagonal
    covariance matrices). mu 2 and logs(var) 2 are optional and will be resp. zeros and zeros if not given.

    A normalization over the batch dimension will automatically be performed.
    An optional normalization over the features dimension can also be performed.

    All tensor sizes should be (N_minibatch, N_channels) """
    def __init__(self, normalize=True):
        self.normalize = normalize  # Normalization over channels

    def __call__(self, mu1, logvar1, mu2=None, logvar2=None):
        if mu2 is None and logvar2 is None:
            Dkl = 0.5 * torch.sum(torch.exp(logvar1) + torch.square(mu1) - logvar1 - 1.0)
        else:
            raise NotImplementedError("General Dkl not implemented yet...")
        Dkl = Dkl / mu1.size(0)
        if self.normalize:
            return Dkl / mu1.size(1)
        else:
            return Dkl


class SynthParamsLossBase:
    def __init__(self, idx_helper: PresetIndexesHelper, compute_symmetrical_presets=False):
        self.idx_helper = idx_helper
        self.compute_symmetrical_presets = compute_symmetrical_presets

    def get_symmetrical_learnable_presets(self, u_out: torch.Tensor, u_in: torch.Tensor) \
            -> (List[range], torch.Tensor, torch.Tensor):
        """ Returns an 'extended batch' with symmetrical presets if self.compute_symmetrical_presets,
         otherwise returns compatible unmodified u_out and u_in.
        See PresetIndexesHelper.get_symmetrical_learnable_presets(...)

        :returns: permutations_groups, u_out_with_duplicates, u_in_with_symmetries
        """
        if self.compute_symmetrical_presets:
            return self.idx_helper.get_symmetrical_learnable_presets(u_out, u_in)
        else:
            return [range(i, i+1) for i in range(u_in.shape[0])], u_out, u_in



class SynthParamsLoss(SynthParamsLossBase):
    """ A 'dynamic' loss which handles different representations of learnable synth parameters
    (numerical and categorical). The appropriate loss can be computed by passing a PresetIndexesHelper instance
    to this class constructor.

    The categorical loss is categorical cross-entropy. """
    def __init__(self, idx_helper: PresetIndexesHelper, normalize_losses: bool, categorical_loss_factor=0.2,
                 prevent_useless_params_loss=True, compute_symmetrical_presets=False,
                 cat_bce=False, cat_softmax=True, cat_softmax_t=1.0,
                 cat_label_smoothing=0.0, target_noise=0.0, cat_use_class_weights=False,
                 dequantized_dense_loss='None',
                 device='cuda:0'):
        """

        :param idx_helper: PresetIndexesHelper instance, created by a PresetDatabase, to convert vst<->learnable params
        :param normalize_losses: If True, losses will be divided by batch size and number of parameters
            in a batch element. If False, losses will only be divided by batch size.
        :param categorical_loss_factor: Factor to be applied to the categorical cross-entropy loss, which is
            much greater than the 'corresponding' MSE loss (if the parameter was learned as numerical)
        :param prevent_useless_params_loss: If True, the class will search for useless params (e.g. params which
            correspond to a disabled oscillator and have no influence on the output sound). This introduces a
            TODO describe overhead here
        :param cat_softmax: Should be set to True if the regression network does not apply softmax at its output.
            This implies that a Categorical Cross-Entropy Loss will be computed on categorical sub-vectors.
        :param cat_softmax_t: Temperature of the softmax activation applied to cat parameters
        :param cat_bce: Binary Cross-Entropy applied to independent outputs (see InverSynth 2019). Very bad
            perfs but option remains available.
        :param dequantized_dense_loss: TODO doc
        """
        super().__init__(idx_helper, compute_symmetrical_presets)
        self.rng = torch.Generator(device=device)
        self.rng.manual_seed(0)
        # Pre-compute indexes lists (to use less CPU). 'num' stands for 'numerical' (not number)
        self.num_indexes = self.idx_helper.get_numerical_learnable_indexes()
        self.cat_indexes = self.idx_helper.get_categorical_learnable_indexes()

        self.normalize_losses = normalize_losses
        if cat_bce and cat_softmax:
            raise ValueError("'cat_bce' (Binary Cross-Entropy) and 'cat_softmax' (implies Categorical Cross-Entropy) "
                             "cannot be both set to True")
        self.dequantized_dense_loss = dequantized_dense_loss
        self.dequantized_criterion = None
        self.cat_bce = cat_bce
        self.cat_softmax = cat_softmax
        self.cat_softmax_t = cat_softmax_t
        self.target_noise = target_noise
        if self.dequantized_dense_loss is not None and self.dequantized_dense_loss != 'None':  # Preempts all others
            self.target_noise = 0  # Dequantization noise cannot be set
            self.cross_entropy_criteria = None
            if self.dequantized_dense_loss.lower() == 'l1':
                self.dequantized_criterion = torch.nn.L1Loss(reduction='none')
            elif self.dequantized_dense_loss.lower() == 'l2' or self.dequantized_dense_loss.lower() == 'mse':
                self.dequantized_criterion = torch.nn.MSELoss(reduction='none')
            else:
                raise ValueError("Dequantized dense loss '{}' is not available".format(self.dequantized_dense_loss))
            # mask remains zero for non-logit outputs
            self.averaging_mask = torch.zeros((1, self.idx_helper.learnable_preset_size), device='cpu')
            for cat_column, cat_learn_indexes in enumerate(self.cat_indexes):
                self.averaging_mask[:, cat_learn_indexes] = 1.0 / len(cat_learn_indexes)
        elif np.isclose(self.cat_softmax_t, 1.0) and not cat_bce:  # Soft T° close to 1.0: directly use pytorch's CE loss
            samples_count = self.idx_helper.cat_params_class_samples_count
            self.cross_entropy_criteria = list()  # 1 criterion per cat-learned synth param
            for vst_idx, learn_model in enumerate(self.idx_helper.vst_param_learnable_model):
                if learn_model == 'cat':
                    # to prevent un-represented labels to get an infinite weight
                    min_sample_count = (30000 // len(self.idx_helper.full_to_learnable[vst_idx])) // 5
                    if cat_use_class_weights:
                        weights = np.maximum(samples_count[vst_idx], min_sample_count)
                        weights = 1.0 / weights
                        weights = torch.from_numpy(weights / weights.mean()).to(device)
                    else:
                        weights = None
                    self.cross_entropy_criteria.append(nn.CrossEntropyLoss(
                        reduction='none', label_smoothing=cat_label_smoothing, weight=weights))
            if not cat_softmax:
                raise AssertionError("Softmax should always be True - the False value is to be deprecated anyway")
        else:
            raise DeprecationWarning("Standard cross entropy is the only available criterion")
            self.cross_entropy_criteria = None
        self.cat_loss_factor = categorical_loss_factor
        self.prevent_useless_params_loss = prevent_useless_params_loss
        # Numerical loss criterion - summation/average over batch dimension is done at the end
        self.numerical_criterion = nn.MSELoss(reduction='none')

    def __call__(self, u_out: torch.Tensor, u_in: torch.Tensor, training=False):
        """ Categorical parameters must be one-hot encoded. This direct __call__ always recomputes permutations
         of presets (if required in the ctor); to avoid this, use the loss_with_permutations function with
         identity permutation groups. """
        permutations_groups, u_out_w_s, u_in_w_s = self.get_symmetrical_learnable_presets(u_out, u_in)
        return self.loss_with_permutations(permutations_groups, u_out_w_s, u_in_w_s, training)

    def loss_with_permutations(self, permutations_groups: List[range], u_out_w_s: torch.Tensor, _u_in_w_s: torch.Tensor,
                               training=False):
        # add noise to targets, if required
        if training and self.target_noise > 0.0:
            u_in_w_s = _u_in_w_s \
                       + torch.normal(mean=0.0, std=self.target_noise, size=_u_in_w_s.shape).to(_u_in_w_s.device)
            u_in_w_s = torch.clamp(u_in_w_s, min=0.0, max=1.0)
        else:
            u_in_w_s = _u_in_w_s
        # At first: we search for useless parameters (whose loss should not be back-propagated)
        useless_num_learn_param_indexes, useless_cat_learn_param_indexes = list(), list()
        if self.prevent_useless_params_loss:  # useless params searched into all presets with permutations
            for row in range(u_in_w_s.shape[0]):
                num_indexes, cat_indexes = self.idx_helper.get_useless_learned_params_indexes(u_in_w_s[row, :])
                useless_num_learn_param_indexes.append(num_indexes)
                useless_cat_learn_param_indexes.append(cat_indexes)

        # - - - numerical loss - - -
        if len(self.num_indexes) > 0:
            # no summation over batch dimension yet
            num_losses = self.numerical_criterion(u_out_w_s[:, self.num_indexes], u_in_w_s[:, self.num_indexes])
            if self.prevent_useless_params_loss:
                # pre-compute the matrix with a few zeros on CPU (many small ops)
                num_losses_factor = torch.ones(num_losses.shape, device='cpu')
                # set loss to 0.0 for disabled parameters (e.g. Dexed operator w/ output level 0.0)
                for row in range(u_in_w_s.shape[0]):
                    for j, num_idx in enumerate(self.num_indexes):  # j is a sub-tensor column index
                        if num_idx in useless_num_learn_param_indexes[row]:
                            num_losses_factor[row, j] = 0.0
                num_losses *= num_losses_factor.to(num_losses.device)  # Apply on GPU using a single op (-1.5s / epoch)
            if self.normalize_losses:
                num_losses = torch.mean(num_losses, dim=1)
            else:
                num_losses = torch.sum(num_losses, dim=1)
        else:
            num_losses = torch.zeros((u_in_w_s.shape[0],), device=u_in_w_s.device)

        # - - - categorical loss - - -

        if self.dequantized_criterion is None:  # - - - (binary) Cross-entropy, ... - - -
            # We'll store all losses in a matrix, then sum all elements of each line to obtain a per-preset cat loss
            cat_losses = torch.zeros((u_in_w_s.shape[0], len(self.cat_indexes)), device=u_in_w_s.device)
            cat_losses_useless_factors = torch.ones((u_in_w_s.shape[0], len(self.cat_indexes)), device='cpu')
            for cat_column, cat_learn_indexes in enumerate(self.cat_indexes):
                # Pre-compute 0-factor for useless params
                if self.prevent_useless_params_loss:  # used at the end - don't backprop cat loss for disabled params
                    for row in range(u_in_w_s.shape[0]):  # Need to check cat index 0 only
                        if cat_learn_indexes[0] in useless_cat_learn_param_indexes[row]:
                            cat_losses_useless_factors[row, cat_column] = 0.0
            # For each categorical output (separate loss computations...)
            for cat_column, cat_learn_indexes in enumerate(self.cat_indexes):
                if self.cross_entropy_criteria is not None:  # Baseline categorical cross-entropy loss
                    # pytorch >= 1.10: target can be given as class probs (lower perf. but allows for noise regul.)
                    target_probabilities = u_in_w_s[:, cat_learn_indexes]
                    cat_losses[:, cat_column] = self.cross_entropy_criteria[cat_column](
                        u_out_w_s[:, cat_learn_indexes], target_probabilities)
                else:  # Other losses
                    # Direct cross-entropy computation. The one-hot target is used to select only q output probabilities
                    # corresponding to target classes with p=1. We only need a limited number of output probabilities
                    # (they actually all depend on each other thanks to the softmax (output) layer).
                    if not self.cat_bce:  # Categorical CE
                        target_one_hot = u_in_w_s[:, cat_learn_indexes].bool()  # Used for tensor-element selection
                    else:  # Binary CE: float values required
                        target_one_hot = u_in_w_s[:, cat_learn_indexes]
                    q_odds = u_out_w_s[:, cat_learn_indexes]  # contains all q odds required for BCE or CCE
                    if not self.cat_bce:  # - - - - - NOT Binary CE => Categorical CE with custom T° - - - - -
                        # softmax T° if required: q_odds might not sum to 1.0 already if no softmax was applied before
                        if self.cat_softmax:  # FIXME use combined softmax and CE for stability, if T° close to 1.0
                            q_odds = torch.softmax(q_odds / self.cat_softmax_t, dim=1)
                        # Then the cross-entropy can be computed (simplified formula thanks to p=1.0 one-hot odds)
                        q_odds = q_odds[target_one_hot]  # CE uses only 1 odd per output vector (thanks to softmax)
                        param_cat_loss = - torch.log(q_odds)
                    else:  # - - - - - Binary Cross-Entropy - - - - -
                        raise AssertionError("BCE is currently disabled")
                        # empirical normalization factor - works quite well to get similar CCE and BCE values
                        param_cat_loss = F.binary_cross_entropy(q_odds, target_one_hot, reduction='mean') / 8.0
                    # CCE and BCE: add the temp loss for the current synth parameter
                    cat_losses[:, cat_column] = param_cat_loss
        else:  # - - - "dequantized" categorical loss - - -
            cat_losses_useless_factors = torch.ones_like(u_in_w_s, device='cpu')
            for cat_column, cat_learn_indexes in enumerate(self.cat_indexes):
                # Pre-compute 0-factor for useless params
                if self.prevent_useless_params_loss:  # used at the end - don't backprop cat loss for disabled params
                    for row in range(u_in_w_s.shape[0]):        # Need to check cat index 0 only
                        if cat_learn_indexes[0] in useless_cat_learn_param_indexes[row]:
                            cat_losses_useless_factors[row, cat_learn_indexes] = 0.0
            # Dequantize all inputs (including numerical inputs, which won't be used)
            # increased noise range to -2.0,0.0 ; 0.0,+2.0
            #    such that the L2 loss gives a larger value if the model outputs the wrong logit
            if not training:
                u_dequant = 2.0 * (u_in_w_s - 0.5)
            else:
                u_dequant = 2.0 * (u_in_w_s - torch.rand(u_in_w_s.shape, generator=self.rng, device=u_in_w_s.device))
            all_logits_loss = self.dequantized_criterion(u_out_w_s, u_dequant)
            # per-synthparam reduction (consider number of output categories)
            self.averaging_mask = self.averaging_mask.to(all_logits_loss.device)
            cat_losses = all_logits_loss * self.averaging_mask.expand(all_logits_loss.shape[0], -1)

        cat_losses *= cat_losses_useless_factors.to(u_in_w_s.device)
        cat_losses = torch.sum(cat_losses, 1)  # Sum all cat losses for each preset

        if self.normalize_losses:  # Normalization vs. number of categorical-learned params
            cat_losses = cat_losses / len(self.cat_indexes)

        # losses weighting - Cross-Entropy is usually be much bigger than MSE. num_loss
        total_losses = num_losses + cat_losses * self.cat_loss_factor  # type: torch.Tensor
        if self.compute_symmetrical_presets:
            # keep only the best loss from each group of symmetrical presets
            best_losses_summed = 0.0
            for preset_idx, perm_r in enumerate(permutations_groups):  # Permutation range
                best_idx = torch.argmin(total_losses[perm_r.start:perm_r.stop])
                best_losses_summed += total_losses[best_idx + perm_r.start]  # Convert perm group idx to minibatch idx
        else:
            best_losses_summed = torch.sum(total_losses)

        # average over batch dimension (always)
        return best_losses_summed / len(permutations_groups)


# TODO this class should not take into account parameters that aren't being used,
#    e.g. parameters of a disabled Dexed oscillator.
#    -----> that's a lot of work, and train/eval times are likely to increase much
class AccuracyAndQuantizedNumericalLoss(SynthParamsLossBase):
    def __init__(self, idx_helper: PresetIndexesHelper,
                 numerical_loss_type='L1', reduce_num_loss=True,
                 reduce_accuracy=True, percentage_accuracy_output=True,
                 limited_vst_params_indexes: Optional[Sequence] = None,
                 compute_symmetrical_presets=False):
        """
        'Quantized' parameters loss: to get a meaningful (but non-differentiable) loss, inferred parameter
        values must be quantized as they would be in the synthesizer.
        Only numerical parameters are involved in the numerical loss computation. The PresetIndexesHelper ctor
        argument allows this class to know which params are numerical.
        The loss to be applied after quantization can be passed as a ctor argument.

        Similarly, provides an Accuracy measurement for categorical parameters.

        This loss breaks the computation path (.backward cannot be applied to it).

        :param idx_helper:
        :param numerical_loss_type: 'L1' or 'MSE'
        :param reduce_accuracy: If True, an averaged accuracy will be returned. If False, a dict of accuracies
            (keys = vst param indexes) is returned.
        :param limited_vst_params_indexes: List of VST params to include into to the loss computation. Can be used
            to measure performance of specific groups of params. Set to None to include all numerical parameters.
        :param compute_symmetrical_presets: If True, the loss of all symmetrical presets will be computed, and the
            lowest loss for each preset will be used.
        """
        super().__init__(idx_helper, compute_symmetrical_presets)
        if numerical_loss_type == 'L1':
            self.numerical_loss = nn.L1Loss(reduction='none')
        elif numerical_loss_type == 'MSE':
            self.numerical_loss = nn.MSELoss(reduction='none')
        else:
            raise ValueError("Unavailable loss '{}'".format(numerical_loss_type))
        self.reduce_num_loss = reduce_num_loss
        self.reduce_accuracy = reduce_accuracy
        self.percentage_accuracy_output = percentage_accuracy_output
        # Cardinality checks
        for vst_idx, _ in self.idx_helper.num_idx_learned_as_cat.items():
            assert self.idx_helper.vst_param_cardinals[vst_idx] > 0
        # Number of parameters considered for this loss (after cat->num, etc... conversions).
        # For tensor pre-alloc and size checks
        self.num_params_count = len(self.idx_helper.num_idx_learned_as_num)\
            + len(self.idx_helper.num_idx_learned_as_cat)
        self.cat_params_count = len(self.idx_helper.cat_idx_learned_as_num)\
            + len(self.idx_helper.cat_idx_learned_as_cat)
        self.limited_vst_params_indexes = limited_vst_params_indexes

    def __call__(self, u_out: torch.Tensor, u_in: torch.Tensor):
        """ Returns the loss for numerical VST params only (searched in u_in and u_out).
        Learnable representations can be numerical (in [0.0, 1.0]) or one-hot categorical.
        The type of representation has been stored in self.idx_helper """
        permutations_groups, u_out_w_s, u_in_w_s = self.get_symmetrical_learnable_presets(u_out, u_in)
        return self.losses_with_permutations(permutations_groups, u_out_w_s, u_in_w_s)

    def losses_with_permutations(self, permutations_groups: List[range],
                                 u_out_w_s: torch.Tensor, u_in_w_s: torch.Tensor):
        # Numerical loss and Accuracy, without any reduction. Don't thread those computations (counter-productive)
        # TODO many small tensor operations: try to compute this on CPU only (and check the difference...)
        all_numerical_losses, num_vst_indexes = self._compute_all_numerical_losses(u_out_w_s, u_in_w_s)
        all_accuracies, acc_vst_indexes = self._compute_all_accuracies(u_out_w_s, u_in_w_s)

        # get the best preset among all available symmetrical presets from a group of permutations
        # TODO This criteria that detects the 'best' result is very arbitrary....
        u_error_scores = torch.sum(all_numerical_losses, dim=1) + (torch.sum(1.0 - all_accuracies, dim=1)) / 2.0
        best_numerical_losses, best_accuracies = list(), list()
        for preset_idx, perm_r in enumerate(permutations_groups):  # preset index (in u_in), Permutation range
            best_idx = torch.argmin(u_error_scores[perm_r.start:perm_r.stop])
            best_idx += perm_r.start  # Convert permutation group index to minibatch index
            best_numerical_losses.append(all_numerical_losses[best_idx:best_idx + 1])
            best_accuracies.append(all_accuracies[best_idx:best_idx + 1])
        best_accuracies = torch.vstack(best_accuracies)
        best_numerical_losses = torch.vstack(best_numerical_losses)

        # handle reduction, or not
        if self.reduce_num_loss:
            num_loss = torch.mean(best_numerical_losses).item()  # Average over all dimensions
        else:  # otherwise, return a dict of per-VST-index mean error
            num_loss = dict()
            for col, vst_idx in enumerate(num_vst_indexes):
                num_loss[vst_idx] = torch.mean(best_numerical_losses[:, col]).item()
        if self.percentage_accuracy_output:
            best_accuracies *= 100.0
        if self.reduce_accuracy:  # Reduction if required
            acc = torch.mean(best_accuracies).item()
        else:  # Otherwise, return a dict of per-VST-index mean accuracy
            acc = dict()
            for col, vst_idx in enumerate(acc_vst_indexes):
                acc[vst_idx] = torch.mean(best_accuracies[:, col]).item()
        return acc, num_loss

    def _compute_all_numerical_losses(self, u_out_w_s: torch.Tensor, u_in_w_s: torch.Tensor):
        """ Computes the non-reduced numerical losses for in/out presets with symmetries (those tensors' dim0 is
        expected to be (much) larger the the original minibatch size). """
        # Partial tensors (for final loss computation) - pre-allocate
        u_in_num = torch.empty((u_in_w_s.shape[0], self.num_params_count), device=u_in_w_s.device, requires_grad=False)
        u_out_num = torch.empty((u_in_w_s.shape[0], self.num_params_count), device=u_in_w_s.device, requires_grad=False)
        # if limited vst indexes: fill with zeros (some allocated cols won't be used). Slow but used for eval only.
        if self.limited_vst_params_indexes is not None:
            u_in_num[:, :], u_out_num[:, :] = 0.0, 0.0
        # Column-by-column tensors filling (parameters' ordering is NOT preserved)
        cur_num_tensors_col = 0
        vst_indexes = np.ones((self.num_params_count, ), dtype=int) * -1
        # quantize numerical learnable representations
        for vst_idx, learn_idx in self.idx_helper.num_idx_learned_as_num.items():
            if self.limited_vst_params_indexes is not None:  # if limited vst indexes:
                if vst_idx not in self.limited_vst_params_indexes:  # continue if this param is not included
                    continue
            param_batch = u_in_w_s[:, learn_idx].detach()
            u_in_num[:, cur_num_tensors_col] = param_batch  # Data copy - does not modify u_in
            param_batch = u_out_w_s[:, learn_idx].detach().clone()
            if self.idx_helper.vst_param_cardinals[vst_idx] > 0:  # don't quantize <0 cardinal (continuous)
                cardinal = self.idx_helper.vst_param_cardinals[vst_idx]
                param_batch = torch.round(param_batch * (cardinal - 1.0)) / (cardinal - 1.0)
            u_out_num[:, cur_num_tensors_col] = param_batch
            vst_indexes[cur_num_tensors_col] = vst_idx
            cur_num_tensors_col += 1
        # convert one-hot encoded learnable representations of (quantized) numerical VST params
        for vst_idx, learn_indexes in self.idx_helper.num_idx_learned_as_cat.items():
            if self.limited_vst_params_indexes is not None:  # if limited vst indexes:
                if vst_idx not in self.limited_vst_params_indexes:  # continue if this param is not included
                    continue
            cardinal = len(learn_indexes)
            # Classes as column-vectors (for concatenation)
            in_classes = torch.argmax(u_in_w_s[:, learn_indexes], dim=-1).detach().type(torch.float)
            u_in_num[:, cur_num_tensors_col] = in_classes / (cardinal-1.0)
            out_classes = torch.argmax(u_out_w_s[:, learn_indexes], dim=-1).detach().type(torch.float)
            u_out_num[:, cur_num_tensors_col] = out_classes / (cardinal-1.0)
            vst_indexes[cur_num_tensors_col] = vst_idx
            cur_num_tensors_col += 1
        # Final size checks
        if self.limited_vst_params_indexes is None:
            if cur_num_tensors_col != self.num_params_count:
                raise AssertionError()
        else:
            pass  # No size check for limited params (a list with unlearned and/or cat params can be provided)
        # Positive diff. if out > in. Non-reduced loss
        return self.numerical_loss(u_out_num, u_in_num), vst_indexes

    def _compute_all_accuracies(self, u_out_w_s: torch.Tensor, u_in_w_s: torch.Tensor):
        """ Returns accuracy (or accuracies) for all categorical VST params. Learnable representations can be numerical
        (in [0.0, 1.0]) or one-hot categorical. The type of representation is stored in self.idx_helper """
        accuracies = torch.empty((u_in_w_s.shape[0], self.cat_params_count),
                                 device=u_in_w_s.device, requires_grad=False)
        # Accuracy of numerical learnable representations (involves quantization)
        # Column-by-column tensors filling (parameters' ordering is NOT preserved)
        cur_num_tensors_col = 0
        # but the list of VST indexes corresponding to each col is also returned
        vst_indexes = np.ones((self.cat_params_count, ), dtype=int) * -1
        for vst_idx, learn_idx in self.idx_helper.cat_idx_learned_as_num.items():
            if self.limited_vst_params_indexes is not None:  # if limited vst indexes:
                if vst_idx not in self.limited_vst_params_indexes:  # continue if this param is not included
                    continue
            cardinal = self.idx_helper.vst_param_cardinals[vst_idx]
            param_batch = torch.unsqueeze(u_in_w_s[:, learn_idx].detach(), 1)  # Column-vector
            # Class indexes, from 0 to cardinal-1
            target_classes = torch.round(param_batch * (cardinal - 1.0)).type(torch.int32)  # New tensor allocated
            param_batch = torch.unsqueeze(u_out_w_s[:, learn_idx].detach(), 1)
            out_classes = torch.round(param_batch * (cardinal - 1.0)).type(torch.int32)  # New tensor allocated
            accuracies[:, cur_num_tensors_col] = (target_classes == out_classes)
            vst_indexes[cur_num_tensors_col] = vst_idx
            cur_num_tensors_col += 1
        # accuracy of one-hot encoded categorical learnable representations
        for vst_idx, learn_indexes in self.idx_helper.cat_idx_learned_as_cat.items():
            if self.limited_vst_params_indexes is not None:  # if limited vst indexes:
                if vst_idx not in self.limited_vst_params_indexes:  # continue if this param is not included
                    continue
            target_classes = torch.argmax(u_in_w_s[:, learn_indexes], dim=-1)  # New tensor allocated
            out_classes = torch.argmax(u_out_w_s[:, learn_indexes], dim=-1)  # New tensor allocated
            accuracies[:, cur_num_tensors_col] = (target_classes == out_classes)
            vst_indexes[cur_num_tensors_col] = vst_idx
            cur_num_tensors_col += 1
        if cur_num_tensors_col != self.cat_params_count or np.any(vst_indexes < 0):
            raise AssertionError()
        return accuracies, vst_indexes



class FlowParamsLoss:
    """
    Estimates the Dkl between the true distribution of synth params p*(v) and the current p_lambda(v) distribution.

    This requires to invert two flows (the regression and the latent flow) in order to estimate the probability of
    some v_in target parameters in the q_Z0(z0) distribution (z0 = invT(invU(v)).
    These invert flows (ideally parallelized) must be provided in the loss constructor
    """
    def __init__(self, idx_helper: PresetIndexesHelper, latent_flow_inverse_function, reg_flow_inverse_function):
        self.idx_helper = idx_helper
        self.latent_flow_inverse_function = latent_flow_inverse_function
        self.reg_flow_inverse_function = reg_flow_inverse_function

    def __call__(self, z_0_mu_logvar, v_target):
        """ Estimate the probability of v_target in the q_Z0(z0) distribution (see details in TODO REF) """

        # FIXME v_target should be "inverse-softmaxed" (because actual output will be softmaxed)

        # TODO apply a factor on categorical params (maybe divide by the size of the one-hot encoded vector?)
        #    how to do that with this inverse flow transform??????

        # Flows reversing - sum of log abs det of inverse Jacobian is used in the loss
        z_K, log_abs_det_jac_inverse_U = self.reg_flow_inverse_function(v_target)
        z_0, log_abs_det_jac_inverse_T = self.latent_flow_inverse_function(z_K)
        # Evaluate q_Z0(z0) (closed-form gaussian probability)
        z_0_log_prob = utils.probability.gaussian_log_probability(z_0, z_0_mu_logvar[:, 0, :], z_0_mu_logvar[:, 1, :])
        # Result is batch-size normalized
        # TODO loss factor as a ctor arg
        return - torch.mean(z_0_log_prob + log_abs_det_jac_inverse_T + log_abs_det_jac_inverse_U) / 1000.0


