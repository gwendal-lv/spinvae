from typing import Optional, List

import warnings

import numpy as np
import torch
import torchinfo
from torch import nn
from torch.distributions.normal import Normal

import config
import model.base
import model.ladderbase
import model.ladderencoder
import model.ladderdecoder
import utils.probability
from data.preset2d import Preset2dHelper
from utils.probability import gaussian_log_probability, standard_gaussian_log_probability, MMD


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
                 x_decoded_proba, x_sampled,
                 u_out, u_numerical_nll, u_categorical_nll, u_l1_error, u_accuracy
                 ):
        """
        Class to store outputs of a hierarchical VAE. Some constructor args are optional depending on the VAE
        architecture and training procedure (e.g. normalizing flows or not, etc...)

        Latent values can be retrieved as N * dimZ tensors (without the latent hierarchy dimension)
        """
        self.z_mu, self.z_var, self.z_sampled= z_mu, z_var, z_sampled
        self.x_decoded_proba, self.x_sampled = x_decoded_proba, x_sampled
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

        :param model_config:
        :param train_config:
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

        # Configuration checks
        if model_config.dim_z > 0:
            warnings.warn("model_config.dim_z cannot be enforced (requested value: {}) and will automatically be set "
                          "by the HierarchicalVAE during its construction.".format(model_config.dim_z))

        # Build encoder and decoder
        self._preset_helper = preset_helper
        if train_config.pretrain_audio_only:
            encoder_opt_args, dummy_u = (None, ) * 3, None
        else:
            assert preset_helper is not None
            encoder_opt_args = (model_config.vae_preset_architecture, model_config.preset_hidden_size, preset_helper)
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
                                train_config.preset_CE_use_weights)
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

    def forward(self, x_target, u_target=None, preset_uids=None, midi_notes=None):
        if u_target is not None and self.pre_training_audio:
            u_target = None  # We force a None value even if a dummy preset was given at input

        # 1) Encode
        z_mu, z_var = self.encoder(x_target, u_target, midi_notes)  # TODO encode

        # 2) Latent values
        # For each latent level, sample z_l from q_phi(z_l|x) using reparametrization trick (no conditional posterior)
        z_sampled = list()
        for lat_lvl in range(len(z_mu)):
            if self.training:  # Sampling in training mode only
                eps = Normal(torch.zeros_like(z_mu[lat_lvl], device=z_mu[lat_lvl].device),
                             torch.ones_like(z_mu[lat_lvl], device=z_mu[lat_lvl].device)).sample()
                z_sampled.append(z_mu[lat_lvl] + torch.sqrt(z_var[lat_lvl]) * eps)
            else:  # eval mode: no random sampling
                z_sampled.append(z_mu[lat_lvl])

        # 3) Decode
        x_decoded_proba, x_sampled, preset_decoder_out = self.decoder(z_sampled, u_target)

        # 4) return all available values using Tensor only, for this method to remain usable
        # with multi-GPU training (mini-batch split over GPUs, all output tensors will be concatenated).
        # This tuple output can be parsed later into a proper HierarchicalVAEOutputs instance.
        out_list = list()
        for lat_lvl in range(len(z_mu)):
            out_list += [z_mu[lat_lvl], z_var[lat_lvl], z_sampled[lat_lvl]]
        out_list += [x_decoded_proba, x_sampled]
        return tuple(out_list) + preset_decoder_out

    def parse_outputs(self, forward_outputs):
        """ Parses tuple output from a self.forward(...) call into a HierarchicalVAEOutputs instance. """
        i = 0
        z_mu, z_var, z_sampled = list(), list(), list()
        for latent_level in range(self.decoder.n_latent_levels):
            z_mu.append(forward_outputs[i])
            z_var.append(forward_outputs[i + 1])
            z_sampled.append(forward_outputs[i + 2])
            i += 3
        x_decoded_proba, x_sampled = forward_outputs[i:i+2]
        i += 2
        # See preset_model.py
        u_out, u_numerical_nll, u_categorical_nll, num_l1_error, acc = forward_outputs[i:i+5]
        assert i+5 == len(forward_outputs)
        return HierarchicalVAEOutputs(
            z_mu, z_var, z_sampled,
            x_decoded_proba, x_sampled,
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

    def vae_loss(self, audio_log_prob_loss, x_shape, ae_out: HierarchicalVAEOutputs):
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
        decoder_out = self._hierarchical_vae.decoder(z_multi_level, None)  # No preset: will return lots of None
        return decoder_out[1]  # Return x_sampled only





if __name__ == "__main__":
    _model_config, _train_config = config.ModelConfig(), config.TrainConfig()
    _model_config.vae_main_conv_architecture = 'specladder8x1_res_swish'
    _model_config.vae_latent_extract_architecture = 'conv_1l_k1x1_gated'
    _model_config.vae_latent_levels = 1
    _model_config.approx_requested_dim_z = 256
    _model_config.vae_preset_architecture = 'tfm_2l_ff_memmlp'
    _model_config.preset_hidden_size = 128
    _model_config.preset_decoder_numerical_distribution = "logistic_mixt3"

    _train_config.pretrain_audio_only = False
    _train_config.minibatch_size = 16
    _train_config.preset_cat_dropout = 0.12
    _train_config.preset_CE_label_smoothing = 0.13
    _train_config.preset_sched_sampling_max_p = 0.0

    config.update_dynamic_config_params(_model_config, _train_config)

    if not _train_config.pretrain_audio_only:
        import data.build
        _ds = data.build.get_dataset(_model_config, _train_config)
        _preset_helper = _ds.preset_indexes_helper
        _dummy_preset = _preset_helper.get_null_learnable_preset(_train_config.minibatch_size)
    else:
        _preset_helper, _dummy_preset = None, None

    hVAE = HierarchicalVAE(_model_config, _train_config, _preset_helper)
    hVAE.train()
    hVAE.set_preset_decoder_scheduled_sampling_p(0.3)  # set to > 0.0 to eval in AR mode
    # hVAE.eval()  # FIXME remove
    vae_out = hVAE(torch.zeros(_model_config.input_audio_tensor_size), _dummy_preset)
    vae_out = hVAE.parse_outputs(vae_out)
    lat_loss, lat_backprop_loss = hVAE.latent_loss(vae_out, 1.0)

    print(hVAE.encoder.get_single_ch_conv_summary())
    print(hVAE.encoder.get_latent_cells_summaries())
    if not _train_config.pretrain_audio_only:
        print(hVAE.encoder.preset_encoder.get_summary(_train_config.minibatch_size))

    print(hVAE.z_shapes)

    print(hVAE.decoder.get_single_ch_conv_summary())
    print(hVAE.decoder.get_latent_cells_summaries())
    if not _train_config.pretrain_audio_only:
        print(hVAE.decoder.preset_decoder.get_summary())

    print(hVAE.get_detailed_summary())

    print(hVAE)

