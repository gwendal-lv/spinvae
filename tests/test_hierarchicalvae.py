import unittest
import copy

import numpy as np
import torch
import torch.nn as nn

import config
from model.hierarchicalvae import HierarchicalVAE


class TestFlattenUnflattenLatentValues(unittest.TestCase):

    def test_flatten_unflatten(self):
        _model_config, _train_config = config.ModelConfig(), config.TrainConfig()
        for _approx_dim_z in [10, 100, 200, 317, 350, 400, 879, 1000, 1500, 2000]:
            if _approx_dim_z >= 1500:
                max_latent_levels = 4
            elif _approx_dim_z >= 350:
                max_latent_levels = 3
            elif _approx_dim_z >= 100:
                max_latent_levels = 2
            else:
                max_latent_levels = 1
            for num_lat_levels in range(1, max_latent_levels + 1):
                _model_config.approx_requested_dim_z = _approx_dim_z
                _model_config.vae_latent_levels = num_lat_levels
                config.update_dynamic_config_params(_model_config, _train_config)
                hVAE = HierarchicalVAE(_model_config, _train_config)
                # Build fake latent tensors with known values
                z_original = list()
                last_idx = 0.0
                for lat_lvl, z_shape in enumerate(hVAE.z_shapes):
                    self.assertEqual(len(z_shape), 4, "4d latent tensors expected (multi-channel 2D feature maps)")
                    z_original.append(torch.empty(z_shape))
                    for n in range(z_shape[0]):
                        for c in range(z_shape[1]):
                            for h in range(z_shape[2]):
                                for w in range(z_shape[3]):
                                    z_original[lat_lvl][n, c, h, w] = last_idx
                                    last_idx += 1.0
                z_flat = hVAE.flatten_latent_values(z_original)
                self.assertEqual(z_flat.shape[1], hVAE.dim_z)
                z_unflattened = hVAE.unflatten_latent_values(z_flat)
                for lat_lvl in range(len(z_original)):
                    self.assertTrue(torch.all(z_unflattened[lat_lvl] == z_original[lat_lvl]))


if __name__ == "__main__":
    unittest.main()

