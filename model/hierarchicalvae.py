from typing import Optional, List

import warnings

import numpy as np
import torch
import torchinfo
from torch.distributions.normal import Normal

import config
import model.base
import model.ladderbase
import model.ladderencoder
import model.ladderdecoder
import utils.probability
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
    def __init__(self, z_mu: List[torch.Tensor], z_var: List[torch.Tensor], z_sampled: List[torch.Tensor], z_loss,
                 x_decoded_proba, x_sampled):
        """
        Class to store outputs of a hierarchical VAE. Some constructor args are optional depending on the VAE
        architecture and training procedure (e.g. normalizing flows or not, etc...)

        Latent values can be retrieved as N * dimZ tensors (without the latent hierarchy dimension)
        """
        self.z_mu, self.z_var, self.z_sampled, self.z_loss = z_mu, z_var, z_sampled, z_loss
        self.x_decoded_proba, self.x_sampled = x_decoded_proba, x_sampled

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
    def __init__(self, model_config: config.ModelConfig, train_config: Optional[config.TrainConfig] = None):
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

        if not train_config.pretrain_audio_only:
            raise NotImplementedError()  # TODO

        self._input_audio_tensor_size = model_config.input_audio_tensor_size

        # Pre-process configuration
        self.main_conv_arch = model.ladderbase.parse_main_conv_architecture(model_config.vae_main_conv_architecture)
        self.latent_arch = parse_latent_extract_architecture(model_config.vae_latent_extract_architecture)

        # Configuration checks
        if model_config.dim_z > 0:
            warnings.warn("model_config.dim_z cannot be enforced (requested value: {}) and will automatically be set "
                          "by the HierarchicalVAE during its construction.".format(model_config.dim_z))

        # Build encoder and decoder
        self.encoder = model.ladderencoder.LadderEncoder(
            self.main_conv_arch, self.latent_arch, model_config.vae_latent_levels, model_config.input_audio_tensor_size,
            approx_dim_z=model_config.approx_requested_dim_z
        )
        with torch.no_grad():  # retrieve encoder output shapes (list of shapes for each latent level)
            dummy_z_mu, _ = self.encoder(torch.zeros(model_config.input_audio_tensor_size))
        self.z_shapes = [z.shape for z in dummy_z_mu]
        model_config.dim_z = self.dim_z  # Compute dim_z and update model_config (.dim_z property uses self.z_shapes)
        if train_config.verbosity >= 1:
            print("[HierarchicalVAE] model_config.dim_z has been automatically set to {} (requested approximate dim_z "
                  "was {})".format(model_config.dim_z, model_config.approx_requested_dim_z))
        self.decoder = model.ladderdecoder.LadderDecoder(
            self.main_conv_arch, self.latent_arch,
            self.z_shapes, model_config.input_audio_tensor_size,
            model_config.audio_decoder_distribution
        )

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

    def get_custom_param_group(self, group_name: str):
        # Convert pytorch's parameters iterator into a list for each model, then sum these lists
        return sum([list(m.get_custom_param_group(group_name)) for m in [self.encoder, self.decoder]], [])

    def forward(self, x, v=None, preset_uids=None, midi_notes=None):
        if v is not None:
            raise NotImplementedError("Presets can not be auto-encoded at the moment (audio input only).")

        # 1) Encode
        z_mu, z_var = self.encoder(x, None, midi_notes)

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
        # We can already compute the per-element latent loss (not batch-averaged/normalized, no beta factor yet)
        # Only the vanilla-VAE Dkl loss is available at the moment
        # FIXME compute Dkl for each latent level, to ensure that the encode approx. the same amount of information
        #    (shallower latents seem to collapse more easily than deeper latents)
        #    HOW TO compute self._dkl_gamma
        z_losses_per_lvl = list()
        for lat_lvl in range(len(z_mu)):
            # Dkl for a batch item is the sum of per-coordinates Dkls
            # We don't normalize (divide by the latent size) to stay closer to the ELBO when the latent size is changed.
            z_losses_per_lvl.append(utils.probability.standard_gaussian_dkl(
                z_mu[lat_lvl].flatten(start_dim=1), z_var[lat_lvl].flatten(start_dim=1), reduction='none'))
        # TODO gamma_l factors applied to KLDs during warmup
        z_loss = sum(z_losses_per_lvl)  # N-dimensional loss (not minibatch-averaged)
        # TODO calculer gamma sur le mini-batch uniquement ?

        # FIXME test
        truc = self.flatten_latent_values(z_sampled)

        # 3) Decode
        x_decoded_proba, x_sampled = self.decoder(z_sampled)

        # 4) return all available values using Tensor only, for this method to remain usable
        # with multi-GPU training (mini-batch split over GPUs, all output tensors will be concatenated).
        # This tuple output can be parsed later into a proper HierarchicalVAEOutputs instance.
        out_list = list()
        for lat_lvl in range(len(z_mu)):
            out_list += [z_mu[lat_lvl], z_var[lat_lvl], z_sampled[lat_lvl]]
        out_list += [z_loss, x_decoded_proba, x_sampled]
        return tuple(out_list)

    def parse_outputs(self, forward_outputs):
        """ Parses tuple output from a self.forward(...) call into a HierarchicalVAEOutputs instance. """
        i = 0
        z_mu, z_var, z_sampled = list(), list(), list()
        for latent_level in range(self.decoder.n_latent_levels):
            z_mu.append(forward_outputs[i])
            z_var.append(forward_outputs[i + 1])
            z_sampled.append(forward_outputs[i + 2])
            i += 3
        return HierarchicalVAEOutputs(
            z_mu, z_var, z_sampled, forward_outputs[i], forward_outputs[i + 1], forward_outputs[i + 2]
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
        batch_latent_loss = torch.mean(ae_out.z_loss)
        return batch_latent_loss, batch_latent_loss * beta

    def vae_loss(self, audio_log_prob_loss, x_shape, ae_out: HierarchicalVAEOutputs):
        """
        Returns a total loss that corresponds to the ELBO if this VAE is a vanilla VAE with Dkl.

        :param audio_log_prob_loss: Mean-reduced log prob loss (averaged over all dimensions)
        :param x_shape: Shape of audio input tensors
        :param ae_out:
        """
        # Factorized distributions - we suppose that the independant log-probs were added
        # We don't consider the  beta factor for the latent loss (but z_loss must be average over the batch dim)
        x_data_dims = np.prod(np.asarray(x_shape[2:]))
        return audio_log_prob_loss * x_data_dims + torch.mean(ae_out.z_loss)

    def get_detailed_summary(self):
        sep_str = '************************************************************************************************\n'
        summary = sep_str + '********** ENCODER audio single-channel conv **********\n' + sep_str
        summary += str(self.encoder.get_single_ch_conv_summary()) + '\n\n'
        summary += sep_str + '********** ENCODER latent cells **********\n' + sep_str
        summary += str(self.encoder.get_latent_cells_summaries()) + '\n\n'
        summary += sep_str + '********** DECODER latent cells **********\n' + sep_str
        summary += str(self.decoder.get_latent_cells_summaries()) + '\n\n'
        summary += sep_str + '********** DECODER audio single-channel conv **********\n' + sep_str
        summary += str(self.decoder.get_single_ch_conv_summary()) + '\n\n'
        summary += sep_str + '********** FULL VAE SUMMARY **********\n' + sep_str
        summary += str(torchinfo.summary(
            self, self._input_audio_tensor_size, depth=5, device=torch.device('cpu'), verbose=0,
            col_names=("input_size", "output_size", "num_params", "mult_adds"),
            row_settings=("depth", "var_names")
        ))
        # TODO also retrieve total number of mult-adds/item and parameters
        return summary



if __name__ == "__main__":
    _model_config, _train_config = config.ModelConfig(), config.TrainConfig()
    _train_config.minibatch_size = 16
    _model_config.vae_main_conv_architecture = 'specladder8x2_res_depsep5x5'
    _model_config.vae_latent_extract_architecture = 'conv_1l_k1x1_gated'
    _model_config.vae_latent_levels = 3
    _model_config.approx_requested_dim_z = 400
    config.update_dynamic_config_params(_model_config, _train_config)

    hVAE = HierarchicalVAE(_model_config, _train_config)

    print(hVAE.encoder.get_single_ch_conv_summary())
    print(hVAE.encoder.get_latent_cells_summaries())

    print(hVAE.decoder.get_single_ch_conv_summary())
    print(hVAE.decoder.get_latent_cells_summaries())

    print(hVAE.get_detailed_summary())

    vae_out = hVAE(torch.zeros(_model_config.input_audio_tensor_size))
    vae_out = hVAE.parse_outputs(vae_out)

    print(hVAE)

