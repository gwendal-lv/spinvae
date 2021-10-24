"""
Classes to generate and evaluate interpolations between samples.
"""

from pathlib import Path
from typing import Tuple

import numpy as np
import scipy.interpolate
import torch

import logs.logger
import model.build
import utils.config


class LatentInterpolationEval:
    def __init__(self, generator, num_steps=7, device='cpu', kind='linear'):
        """

        :param generator: Any model that provides a generate_from_latent_vector(z) method (where z is a batch of
            vectors), a .dim_z attribute, and... ? TODO finish doc
        :param kind: See https://docs.scipy.org/doc/scipy/reference/reference/generated/scipy.interpolate.interp1d.html
        """
        self.gen = generator
        self.num_steps = num_steps
        self.device = device
        self.kind = kind

    @property
    def dim_z(self):
        return self.gen.dim_z

    def interpolate_spectrograms_from_latent(self, z_start, z_end) -> Tuple[np.ndarray, torch.Tensor, torch.Tensor]:
        """ Returns u, z, and a N x 1 x H x W tensor of interpolated spectrograms, using the latent representations
            of those spectrograms as interpolation inputs. """
        u, z = self.interpolate_latent(z_start, z_end)
        return u, z, self.gen.generate_from_latent_vector(z)

    def interpolate_latent(self, z_start, z_end) -> Tuple[np.ndarray, torch.Tensor]:
        """ Returns a N x D tensor of interpolated latent vectors, where N is the number of interpolation steps (here:
        considered as a batch size) and D is the latent dimension. Each latent coordinate is interpolated independently.

        Non-differentiable: based on scipy.interpolate.interp1d.

        :param z_start: 1 x D tensor
        :param z_end: 1 x D tensor
        :returns: u, interpolated_z
        """
        # TODO try RBF interpolation instead of interp 1d ?
        z_cat = torch.cat([z_start, z_end], dim=0)
        interp_f = scipy.interpolate.interp1d([0.0, 1.0], z_cat.clone().detach().cpu().numpy(), kind=self.kind, axis=0,
                                              bounds_error=True)  # extrapolation disabled, no fill_value
        u_interpolated = np.linspace(0.0, 1.0, self.num_steps, endpoint=True)
        return u_interpolated, torch.tensor(interp_f(u_interpolated), device=self.device, dtype=torch.float32)

    # TODO generate and save independent figure (easier to show in HTML tables // github pages)

    # TODO compute 'interpolation smoothness' coefficients


if __name__ == "__main__":
    import evaluation.load
    _device = 'cpu'
    model_loader = evaluation.load.ModelLoader("saved/MMD_tests/mmd_determ_enc_lossx5_drop0.3_wd_1e-4", _device)

    interpolator = LatentInterpolationEval(model_loader.ae_model, device=_device)
    _z_start, _z_end = torch.ones((1, interpolator.dim_z)) * 0.0, torch.ones((1, interpolator.dim_z)) * 1.0
    #_z_start, _z_end = torch.normal(0.0, 1.0, (1, interpolator.dim_z)), torch.normal(0.0, 1.0, (1, interpolator.dim_z))
    _u, _z, _x = interpolator.interpolate_spectrograms_from_latent(_z_start, _z_end)

    print(model_loader.model_config)
