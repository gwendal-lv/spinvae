from typing import Optional, List, Dict, Union, Any

import warnings

import numpy as np
import torch
import torchinfo
from torch import nn
import torch.nn.functional as F
from torch.distributions.normal import Normal

import config
import model.base
import model.ladderbase
import model.ladderencoder
import model.ladderdecoder
import utils.probability
from data.preset2d import Preset2dHelper
from utils.probability import gaussian_log_probability, standard_gaussian_log_probability, MMD
import utils.exception


def parse_latent_extract_architecture(full_architecture: str):
    arch_args = full_architecture.split('_')
    base_arch_name = arch_args[0].lower()
    num_layers = int(arch_args[1].replace('l', ''))
    arch_args_dict = {
        'k1x1': False, 'k3x3': False,  # Only these 2 kernel sizes are available
        'posenc': False,  # Positional encodings can be added inside some architectures
        'gated': False,  # (Self-)gating ("light attention") mechanisms can be added to some architectures
        'att': False,  # SAGAN-like self-attention
    }
    for arch_arg in arch_args[2:]:
        if arch_arg in arch_args_dict.keys():
            arch_args_dict[arch_arg] = True  # Authorized arguments
        else:
            raise ValueError("Unvalid encoder argument '{}' from architecture '{}'".format(arch_arg, full_architecture))
    return {'name': base_arch_name, 'n_layers': num_layers, 'args': arch_args_dict}


class HierarchicalVAEOutputs:
    def __init__(self, z_mu: List[torch.Tensor], z_var: List[torch.Tensor], z_sampled: List[torch.Tensor],
                 x_decoded_proba, x_sampled, x_target_NLL,
                 u_out, u_numerical_nll, u_categorical_nll, u_l1_error, u_accuracy
                 ):
        """
        Class to store outputs of a hierarchical VAE. Some constructor args are optional depending on the VAE
        architecture and training procedure (e.g. normalizing flows or not, etc...)

        Latent values can be retrieved as N * dimZ tensors (without the latent hierarchy dimension)
        """
        self.z_mu, self.z_var, self.z_sampled= z_mu, z_var, z_sampled
        self.x_decoded_proba, self.x_sampled, self.x_target_NLL = x_decoded_proba, x_sampled, x_target_NLL
        self.u_out, self.u_numerical_nll, self.u_categorical_nll, self.u_l1_error, self.u_accuracy = \
            u_out, u_numerical_nll, u_categorical_nll, u_l1_error, u_accuracy
        # Quick tests to try to ensure that the proper data was provided
        if u_out is not None:
            assert len(u_out.shape) == 3
            assert len(u_numerical_nll.shape) == 1
            assert len(u_accuracy.shape) == 1
        # Can be assigned later
        self.z_loss = None

    def get_z_mu_no_hierarchy(self, to_numpy=False):
        flat_t = torch.cat([torch.flatten(z, start_dim=1) for z in self.z_mu], dim=1)
        return flat_t if not to_numpy else self._to_numpy(flat_t)

    def get_z_var_no_hierarchy(self, to_numpy=False):
        flat_t = torch.cat([torch.flatten(z, start_dim=1) for z in self.z_var], dim=1)
        return flat_t if not to_numpy else self._to_numpy(flat_t)

    def get_z_sampled_no_hierarchy(self, to_numpy=False):
        flat_t = torch.cat([torch.flatten(z, start_dim=1) for z in self.z_sampled], dim=1)
        return flat_t if not to_numpy else self._to_numpy(flat_t)

    @staticmethod
    def _to_numpy(t):
        return t.clone().detach().cpu().numpy()


class HierarchicalVAE(model.base.TrainableMultiGroupModel):
    def __init__(self, model_config: config.ModelConfig, train_config: Optional[config.TrainConfig] = None,
                 preset_helper: Optional[Preset2dHelper] = None):
        """
        Builds a Hierarchical VAE which encodes and decodes multi-channel spectrograms or waveforms.
        A preset can be encoded/decoded as well (optional during pre-training).

        The hierarchical structures allows to infer latent vectors from different levels, not only from
        the top level (vanilla VAE).
        The encoder is based on a single-channel CNN build upon multiple cells. Each cell's output can be used to
        infer a partial latent vector.
        """
        trainable_param_group_names = ['audio', 'latent'] + ([] if train_config.pretrain_audio_only else ['preset'])
        super().__init__(train_config, ['audio', 'latent', 'preset'], trainable_param_group_names)

        # Save some important values from configurations
        self._input_audio_tensor_size = model_config.input_audio_tensor_size
        self._dkl_auto_gamma = train_config.dkl_auto_gamma
        self._latent_free_bits = train_config.latent_free_bits
        self.beta_warmup_ongoing = train_config.beta_warmup_epochs > 0  # Will be modified when warmup has ended

        # Pre-process configuration
        self.main_conv_arch = model.ladderbase.parse_main_conv_architecture(model_config.vae_main_conv_architecture)
        self.latent_arch = parse_latent_extract_architecture(model_config.vae_latent_extract_architecture)
        self.preset_ae_method = model_config.preset_ae_method
        self._alignment_criterion = train_config.preset_alignment_criterion

        # Configuration checks
        if model_config.dim_z > 0:
            warnings.warn("model_config.dim_z cannot be enforced (requested value: {}) and will automatically be set "
                          "by the HierarchicalVAE during its construction.".format(model_config.dim_z))
        if not train_config.pretrain_audio_only:
            if self.preset_ae_method not in ['no_encoding', 'combined_vae', 'no_audio', 'aligned_vaes']:
                raise ValueError("preset_ae_method '{}' not available".format(self.preset_ae_method))
            if self.preset_ae_method == 'aligned_vaes':
                assert model_config.vae_preset_encode_add == 'after_latent_cell', \
                    "Independent VAEs should not share the latent cells "
        else:
            assert self.preset_ae_method is None

        # Build encoder and decoder
        self._preset_helper = preset_helper
        if train_config.pretrain_audio_only:
            encoder_opt_args, dummy_u = (None, ) * 5, None
        else:
            assert preset_helper is not None
            encoder_opt_args = (model_config.vae_preset_architecture,
                                model_config.preset_hidden_size,
                                model_config.vae_preset_encode_add,
                                preset_helper,
                                train_config.preset_internal_dropout)
            dummy_u = preset_helper.get_null_learnable_preset(train_config.minibatch_size)
        self.encoder = model.ladderencoder.LadderEncoder(
            self.main_conv_arch, self.latent_arch, model_config.vae_latent_levels, model_config.input_audio_tensor_size,
            model_config.approx_requested_dim_z,
            *encoder_opt_args
        )
        with torch.no_grad():  # retrieve encoder output shapes (list of shapes for each latent level)
            dummy_z_mu, _ = self.encoder(torch.zeros(model_config.input_audio_tensor_size), dummy_u)
        self.z_shapes = [z.shape for z in dummy_z_mu]
        model_config.dim_z = self.dim_z  # Compute dim_z and update model_config (.dim_z property uses self.z_shapes)
        if train_config.verbosity >= 1:
            print("[HierarchicalVAE] model_config.dim_z has been automatically set to {} (requested approximate dim_z "
                  "was {})".format(model_config.dim_z, model_config.approx_requested_dim_z))
            print("[HierarchicalVAE] latent feature maps shapes: {}".format([s[1:] for s in self.z_shapes]))
            print("[HierarchicalVAE] dim_z for each latent level: {}".format([np.prod(s[1:]) for s in self.z_shapes]))
        if train_config.pretrain_audio_only:
            decoder_opt_args = (None, ) * 7
        else:
            decoder_opt_args = (model_config.vae_preset_architecture,
                                model_config.preset_hidden_size,
                                model_config.preset_decoder_numerical_distribution,
                                preset_helper,
                                self.encoder.preset_encoder.embedding,  # Embedding net is built by the encoder
                                train_config.preset_internal_dropout, train_config.preset_cat_dropout,
                                train_config.preset_CE_label_smoothing,
                                train_config.preset_CE_use_weights,
                                train_config.params_loss_exclude_useless)
        self.decoder = model.ladderdecoder.LadderDecoder(
            self.main_conv_arch, self.latent_arch,
            self.z_shapes, model_config.input_audio_tensor_size,
            model_config.audio_decoder_distribution,
            *decoder_opt_args
        )

        # Build a ModuleList for each group of parameters (e.g. audio/latent/preset, ...)
        # Their only use is to aggregate parameters into a single nn.Module
        self._aggregated_modules_lists = {
            k : nn.ModuleList([m.get_custom_group_module(k) for m in [self.encoder, self.decoder]])
            for k in self.param_group_names
        }

        # Losses and metrics
        self.mmd = MMD()

        # Optimizers and schedulers (all sub-nets, param groups must have been created at this point)
        self._init_optimizers_and_schedulers()

    @property
    def dim_z_per_level(self):
        return [int(np.prod(z_shape[1:])) for z_shape in self.z_shapes]  # cast to int for JSON serialization

    @property
    def dim_z(self):
        return sum(self.dim_z_per_level)

    def get_custom_group_module(self, group_name: str) -> nn.Module:
        return self._aggregated_modules_lists[group_name]

    def forward(self, x_target, u_target=None, preset_uids=None, midi_notes=None, pass_index=0):
        # TODO handle all cases: auto synth prog, preset auto-encoding, ...
        if pass_index == 0:  # - - - - - 1st pass (might be the single one) - - - - -
            if self.pre_training_audio or self.preset_ae_method == 'no_encoding':
                x_input, u_input = x_target, None
            elif self.preset_ae_method == 'combined_vae':
                x_input, u_input = x_target, u_target
            elif self.preset_ae_method == 'aligned_vaes':
                x_input, u_input = None, u_target  # 1st default pass: preset (s.t. default always outputs a preset)
                x_target = None  # will disable computation of x outputs (audio distribution and samples)
            elif self.preset_ae_method == 'no_audio':
                x_input, u_input = None, u_target
                x_target = None  # will disable computation of x outputs (audio distribution and samples)
            else:
                raise NotImplementedError(self.preset_ae_method)
        elif pass_index == 1:  # - - - - -  2nd pass (optional) - - - - -
            if self.preset_ae_method == 'aligned_vaes':
                x_input, u_input = x_target, None  # 2nd pass: audio only
                u_target = None  # disable preset loss computation
            else:
                raise AssertionError("This model's training procedure is single-pass.")
        else:
            raise ValueError(pass_index)

        # Encode, sample, decode
        z_mu, z_var = self.encoder(x_input, u_input, midi_notes)
        z_sampled = self.sample_z(z_mu, z_var)
        # TODO maybe don't use x_target if auto-encoding preset only
        x_decoded_proba, x_sampled, x_target_NLL, preset_decoder_out = self.decoder(
            z_sampled,
            u_target=u_target,
            x_target=x_target, compute_x_out=(x_target is not None)
        )

        # Outputs: return all available values using Tensor only, for this method to remain usable
        #     with multi-GPU training (mini-batch split over GPUs, all output tensors will be concatenated).
        #     This tuple output can be parsed later into a proper HierarchicalVAEOutputs instance.
        #     All of these output tensors must retain the batch dimension
        out_list = list()
        for lat_lvl in range(len(z_mu)):
            out_list += [z_mu[lat_lvl], z_var[lat_lvl], z_sampled[lat_lvl]]
        out_list += [x_decoded_proba, x_sampled, x_target_NLL]
        return tuple(out_list) + preset_decoder_out

    def sample_z(self, z_mu, z_var):
        # For each latent level, sample z_l from q_phi(z_l|x) using reparametrization trick (no conditional posterior)
        z_sampled = list()
        for lat_lvl in range(len(z_mu)):
            if self.training:  # Sampling in training mode only
                eps = Normal(torch.zeros_like(z_mu[lat_lvl], device=z_mu[lat_lvl].device),
                             torch.ones_like(z_mu[lat_lvl], device=z_mu[lat_lvl].device)).sample()
                z_sampled.append(z_mu[lat_lvl] + torch.sqrt(z_var[lat_lvl]) * eps)
            else:  # eval mode: no random sampling
                z_sampled.append(z_mu[lat_lvl])
        return z_sampled

    def parse_outputs(self, forward_outputs):
        """ Parses tuple output from a self.forward(...) call into a HierarchicalVAEOutputs instance. """
        i = 0
        z_mu, z_var, z_sampled = list(), list(), list()
        for latent_level in range(self.decoder.n_latent_levels):
            z_mu.append(forward_outputs[i])
            z_var.append(forward_outputs[i + 1])
            z_sampled.append(forward_outputs[i + 2])
            i += 3
        x_decoded_proba, x_sampled, x_target_NLL = forward_outputs[i:i+3]
        i += 3
        # See preset_model.py
        u_out, u_numerical_nll, u_categorical_nll, num_l1_error, acc = forward_outputs[i:i+5]
        assert i+5 == len(forward_outputs)

        return HierarchicalVAEOutputs(
            z_mu, z_var, z_sampled,
            x_decoded_proba, x_sampled, x_target_NLL,
            u_out, u_numerical_nll, u_categorical_nll, num_l1_error, acc
        )

    @staticmethod
    def flatten_latent_values(z_multi_level: List[torch.Tensor]):
        """ Flattens all dimensions but the batch dimension. """
        return torch.cat([z.flatten(start_dim=1) for z in z_multi_level], dim=1)

    def unflatten_latent_values(self, z_flat: torch.Tensor):
        """
        Transforms a N x dimZ flatten latent vector into a List of N x Ci x Hi x Wi tensors, where
            Ci, Hi, Wi is the shape a latent feature map for latent level i.
        """
        # Split the flat tensor into 1 group / latent level (known shapes)
        z_flat_per_level = torch.split(z_flat, self.dim_z_per_level, dim=1)
        # Actual unflatten here - unittested to ensure that this operation perfectly reverses flatten_latent_values
        z_unflattened_per_level = list()
        for lat_lvl, z_lvl_flat in enumerate(z_flat_per_level):
            # + operator: concatenates torch Sizes
            z_unflattened_per_level.append(z_lvl_flat.view(*(z_flat.shape[0:1] + self.z_shapes[lat_lvl][1:])))
        return z_unflattened_per_level

    def latent_loss(self, ae_out: HierarchicalVAEOutputs, beta):
        """ Returns the non-normalized latent loss, and the same value multiplied by beta (for backprop) """
        # FIXME for hybrid training (parallel preset/audio VAEs), use a different beta for different batch items
        #    (if possible? not batch-reduced yet?)
        z_mu, z_var = ae_out.z_mu, ae_out.z_var
        # Only the vanilla-VAE Dkl loss is available at the moment
        z_losses_per_lvl = list()  # Batch-averaged losses
        for lat_lvl in range(len(z_mu)):
            # Dkl for a batch item is the sum of per-coordinates Dkls
            # We don't normalize (divide by the latent size) to stay closer to the ELBO when the latent size is changed.
            if np.isclose(self._latent_free_bits, 0.0):
                z_losses_per_lvl.append(utils.probability.standard_gaussian_dkl(
                    z_mu[lat_lvl].flatten(start_dim=1), z_var[lat_lvl].flatten(start_dim=1), reduction='mean'))
            else:
                # "Free bits" constraint is applied to each channel
                # The hyper-param is given as a "per-pixel" free bits (for it to be generalized to any
                # latent feature map size)
                min_dkl = torch.tensor(
                    self._latent_free_bits * np.prod(z_mu[lat_lvl].shape[2:]),
                    dtype=z_mu[lat_lvl].dtype, device=z_mu[lat_lvl].device)
                # Average over batch dimension (not over pixels dimensions), but keep the channels dim
                dkl_per_ch = utils.probability.standard_gaussian_dkl_2d(
                    z_mu[lat_lvl], z_var[lat_lvl], dim=(2, 3), reduction='mean')
                dkl_per_ch = torch.maximum(dkl_per_ch, min_dkl)  # Free-bits constraint
                # Sum Dkls from all channels (overall diagonal gaussian prior)
                z_losses_per_lvl.append(torch.sum(dkl_per_ch))
        # Compute a Dkl factor for each latent level, to ensure that they encode approx. the same amount
        #    of information (shallower latents seem to collapse more easily than deeper latents)?
        #    These gamma_l factors can be applied to KLDs during warmup (and if activated) - then optimize (beta-)ELBO
        if self._dkl_auto_gamma and self.beta_warmup_ongoing:
            with torch.no_grad():  # 0.5 arbitrary factor - TODO try to find a better solution
                dkl_per_level = 0.5 * np.asarray([_z_loss.item() for _z_loss in z_losses_per_lvl])
            dkl_gamma_per_level = dkl_per_level / dkl_per_level.mean()
            z_loss = sum([_z_loss * dkl_gamma_per_level[lvl] for lvl, _z_loss in enumerate(z_losses_per_lvl)])
        else:
            z_loss = sum([_z_loss for _z_loss in z_losses_per_lvl])

        # Store the new loss in the ae_out structure
        ae_out.z_loss = z_loss
        return z_loss, z_loss * beta

    def latent_alignment_loss(self, ae_out_audio: HierarchicalVAEOutputs, ae_out_preset: HierarchicalVAEOutputs, beta):
        """ Returns a loss that should be minimized to better align two parallel VAEs (one preset-VAE, one audio-VAE),
            or returns 0.0 if self.preset_ae_method != 'aligned_vaes' """
        if self.preset_ae_method != 'aligned_vaes':
            assert self._alignment_criterion is None
            return 0.0
        else:
            mu_audio = self.flatten_latent_values(ae_out_audio.z_mu)
            var_audio = self.flatten_latent_values(ae_out_audio.z_var)
            mu_preset = self.flatten_latent_values(ae_out_preset.z_mu)
            var_preset = self.flatten_latent_values(ae_out_preset.z_var)
            if self._alignment_criterion == 'kld':  # KLD ( q(z|preset) || q(z|audio) )
                loss = utils.probability.gaussian_dkl(mu_preset, var_preset, mu_audio, var_audio, reduction='mean')
            elif self._alignment_criterion == 'symmetric_kld':
                loss = utils.probability.symmetric_gaussian_dkl(
                    mu_preset, var_preset, mu_audio, var_audio, reduction='mean')
            # TODO implement 2-Wasserstein distance (and MMD?)
            else:
                raise NotImplementedError(self._alignment_criterion)
        return loss * beta

    def audio_vae_loss(self, audio_log_prob_loss, x_shape, ae_out: HierarchicalVAEOutputs):
        """
        Returns a total loss that corresponds to the ELBO if this VAE is a vanilla VAE with Dkl.

        :param audio_log_prob_loss: Mean-reduced log prob loss (averaged over all dimensions)
        :param x_shape: Shape of audio input tensors
        :param ae_out:
        """
        # Factorized distributions - we suppose that the independant log-probs were added
        # We don't consider the  beta factor for the latent loss (but z_loss must be average over the batch dim)
        x_data_dims = np.prod(np.asarray(x_shape[1:]))  # C spectrograms of size H x W
        return audio_log_prob_loss * x_data_dims + ae_out.z_loss

    def set_preset_decoder_scheduled_sampling_p(self, p: float):
        if self.decoder.preset_decoder is not None:  # Does not send a warning if preset decoder does not exist
            self.decoder.preset_decoder.child_decoder.scheduled_sampling_p = p

    def get_detailed_summary(self):
        sep_str = '************************************************************************************************\n'
        summary = sep_str + '********** ENCODER audio single-channel conv **********\n' + sep_str
        summary += str(self.encoder.get_single_ch_conv_summary()) + '\n\n'
        if self.encoder.preset_encoder is not None:
            summary += sep_str + '********** ENCODER preset **********\n' + sep_str
            summary += str(self.encoder.preset_encoder.get_summary(self._input_audio_tensor_size[0])) + '\n\n'
        summary += sep_str + '********** ENCODER latent cells **********\n' + sep_str
        summary += str(self.encoder.get_latent_cells_summaries()) + '\n\n'
        summary += sep_str + '********** DECODER latent cells **********\n' + sep_str
        summary += str(self.decoder.get_latent_cells_summaries()) + '\n\n'
        summary += sep_str + '********** DECODER audio single-channel conv **********\n' + sep_str
        summary += str(self.decoder.get_single_ch_conv_summary()) + '\n\n'
        if self.decoder.preset_decoder is not None:
            summary += sep_str + '********** DECODER preset **********\n' + sep_str
            summary += str(self.decoder.preset_decoder.get_summary()) + '\n\n'
        summary += sep_str + '********** FULL VAE SUMMARY **********\n' + sep_str
        if self._preset_helper is None:
            input_data = (torch.rand(self._input_audio_tensor_size) - 0.5, )
        else:
            input_u = self._preset_helper.get_null_learnable_preset(self._input_audio_tensor_size[0])
            input_data = (torch.rand(self._input_audio_tensor_size) - 0.5, input_u)
        summary += str(torchinfo.summary(
            self, input_data=input_data, depth=5, device=torch.device('cpu'), verbose=0,
            col_names=("input_size", "output_size", "num_params", "mult_adds"),
            row_settings=("depth", "var_names")
        ))
        # TODO also retrieve total number of mult-adds/item and parameters
        return summary


def process_minibatch(
        ae_model: HierarchicalVAE, ae_model_parallel: nn.DataParallel, main_device,
        x_in, v_in, uid, notes, label,
        epoch: int, scalars: Dict[str, Any], super_metrics: Dict[str, Any]
):
    training = ae_model.training
    suffix = "/Train" if training else "/Valid"

    if training:
        ae_model.optimizers_zero_grad()
    # 1- or 2-pass forward
    ae_out_0 = ae_model_parallel(x_in, v_in, uid, notes, pass_index=0)
    ae_out_0 = ae_model.parse_outputs(ae_out_0)
    if ae_model.preset_ae_method == 'aligned_vaes':
        ae_out_1 = ae_model_parallel(x_in, v_in, uid, notes, pass_index=1)
        ae_out_1 = ae_model.parse_outputs(ae_out_1)
    else:
        ae_out_1 = None
    # assign the output(s) to preset- and/or audio-outputs (not to log Nones, and to log the proper values)
    if ae_model.preset_ae_method in ['no_encoding', 'combined_vae']:
        ae_out_audio, ae_out_preset = ae_out_0, ae_out_0
    elif ae_model.preset_ae_method == 'no_audio':
        ae_out_audio, ae_out_preset = None, ae_out_0
    elif ae_model.preset_ae_method == 'aligned_vaes':
        ae_out_audio, ae_out_preset = ae_out_1, ae_out_0
    else:
        raise NotImplementedError()

    # always log "preset-related" latent metrics (in the end, we'll be interested in presets only)
    #    they are often the same as "audio-related" latent metrics (also true during pre-training without presets)
    super_metrics['LatentMetric' + suffix].append_hierarchical_latent(ae_out_preset, label)

    # Losses (some of them are computed on 1 GPU using the non-parallel original model instance)
    beta = scalars['Sched/VAE/beta'].get(epoch)
    if ae_out_audio is not None:
        audio_log_prob_loss = ae_out_audio.x_target_NLL.mean()
        scalars['Audio/LogProbLoss' + suffix].append(audio_log_prob_loss)
    else:
        audio_log_prob_loss = torch.zeros((1,), device=main_device)
    # TODO which AE_OUT to use for 2-pass models???
    #    compute both, then average?
    #    use a different beta
    lat_loss, lat_backprop_loss = ae_model.latent_loss(ae_out_0, beta)
    if ae_out_1 is not None:
        lat_loss_1, lat_backprop_loss_1 = ae_model.latent_loss(ae_out_1, beta)
        lat_loss += lat_loss_1  # don't average - would be equivalent to reducing beta for each VAE
        lat_backprop_loss += lat_backprop_loss_1
    scalars['Latent/Loss' + suffix].append(lat_loss)
    scalars['Latent/BackpropLoss' + suffix].append(lat_backprop_loss)  # Includes beta
    align_loss = ae_model.latent_alignment_loss(ae_out_audio, ae_out_preset, beta)
    scalars['Latent/AlignLoss' + suffix].append(align_loss)
    if not ae_model.pre_training_audio:
        # TODO use a ae_out_preset for this ?
        u_categorical_nll, u_numerical_nll = ae_out_preset.u_categorical_nll.mean(), ae_out_preset.u_numerical_nll.mean()
        preset_loss = u_categorical_nll + u_numerical_nll
        scalars['Preset/NLL/Total' + suffix].append(preset_loss)
        preset_loss *= ae_model.params_loss_compensation_factor
        scalars['Preset/NLL/Numerical' + suffix].append(u_numerical_nll)
        scalars['Preset/NLL/CatCE' + suffix].append(u_categorical_nll)
    else:
        preset_loss = torch.zeros((1,), device=main_device)
    # FIXME training or not
    preset_reg_loss = torch.zeros((1,), device=main_device)  # No regularization yet....

    with torch.no_grad():  # Monitoring-only losses
        scalars['Latent/MMD' + suffix].append(ae_model.mmd(ae_out_preset.get_z_sampled_no_hierarchy()))
        if not ae_model.pre_training_audio:
            scalars['Preset/Accuracy' + suffix].append(ae_out_preset.u_accuracy.mean())
            scalars['Preset/L1error' + suffix].append(ae_out_preset.u_l1_error.mean())
        if ae_out_audio is not None:
            scalars['Audio/MSE' + suffix].append(F.mse_loss(ae_out_audio.x_sampled, x_in))
            scalars['VAELoss/Total' + suffix].append(ae_model.audio_vae_loss(
                audio_log_prob_loss, x_in.shape, ae_out_audio))
            scalars['VAELoss/Backprop' + suffix].append(audio_log_prob_loss + lat_backprop_loss)

    if training:
        utils.exception.check_nan_values(
            epoch, audio_log_prob_loss, lat_backprop_loss, align_loss, preset_loss, preset_reg_loss)
        # Backprop and optimizers' step (before schedulers' step)
        (audio_log_prob_loss + lat_backprop_loss + align_loss + preset_loss + preset_reg_loss).backward()
        ae_model.optimizers_step()

    return ae_out_audio, ae_out_preset


# FIXME move to a different .py file?
class AudioDecoder:
    def __init__(self, hierachical_vae: HierarchicalVAE):
        """ A simple wrapper class for a HierarchicalVAE instance to be used by an
         evaluation.interp.LatentInterpolation instance. """
        self._hierarchical_vae = hierachical_vae

    @property
    def dim_z(self):
        return self._hierarchical_vae.dim_z

    def generate_from_latent_vector(self, z):
        # input z is expected to be a flattened tensor
        z_multi_level = self._hierarchical_vae.unflatten_latent_values(z)
        # No preset: will return lots of None (or zeroes, maybe...)
        decoder_out = self._hierarchical_vae.decoder(z_multi_level, x_target=None, u_target=None)
        return decoder_out[1]  # Return x_sampled only

