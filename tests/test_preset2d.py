
import unittest

import numpy as np

import config
import data.dataset
from data.dataset import DexedDataset
from data.preset2d import Preset2dHelper, Preset2d


"""
model_config, train_config = config.ModelConfig(), config.TrainConfig()
    config.update_dynamic_config_params(model_config, train_config)

    operators = model_config.dataset_synth_args[1]
    # continuous_params_max_resolution = 100

    # No label restriction, no normalization, etc...
    dexed_dataset = DexedDataset(
        ** dataset.model_config_to_dataset_kwargs(model_config),
        algos=None,  # allow all algorithms
        operators=operators,  # Operators limitation (config.py, or chosen above)
        restrict_to_labels=None,
        check_constrains_consistency=(not regen_wav) and (not regen_spectrograms)
    )
"""


class Presets2dTest(unittest.TestCase):

    def test_raw_to_tensor_conversion(self):
        model_config, train_config = config.ModelConfig(), config.TrainConfig()
        config.update_dynamic_config_params(model_config, train_config)
        dexed_dataset = DexedDataset(
            **data.dataset.model_config_to_dataset_kwargs(model_config),
            check_constrains_consistency=False
        )
        for uid in dexed_dataset.valid_preset_UIDs:
            for preset_var in range(dexed_dataset._nb_preset_variations_per_note):
                # Get a Preset2d from the dataset (no direct access to raw np arrays)
                original_preset2d = dexed_dataset.get_full_preset_params(uid, preset_var)
                self.assertTrue(original_preset2d.is_from_raw_preset)
                raw_original_preset = original_preset2d._raw_preset
                # Convert to tensor and back to numpy arrays
                new_preset2d = Preset2d(dexed_dataset, learnable_tensor_preset=original_preset2d.to_learnable_tensor())
                raw_new_preset = new_preset2d.to_raw()
                # Resulting np array should be exactly the same
                self.assertTrue(
                    np.all(np.isclose(raw_new_preset, raw_original_preset, atol=1e-5)),
                    "{} and {} are not np.isclose (difference = {})"
                        .format(raw_new_preset, raw_original_preset, raw_new_preset - raw_original_preset)
                )


if __name__ == "__main__":
    unittest.main()
