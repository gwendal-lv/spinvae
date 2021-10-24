import unittest
from pathlib import Path
import pickle
import copy

import numpy as np

from data.dataset import DexedDataset
import model.loss


class TestDexedDatasetLoader:
    def __init__(self, force_refresh=False, vst_params_learned_as_categorical='all<=32'):
        # If True, dataset will be re-created and re-stored (quite long, should be done once)
        self.refresh_pickled_dataset = force_refresh
        self.params_learned_as_cat = vst_params_learned_as_categorical
        self.dataset = None

    @property
    def _dataset_pickle_filepath(self):
        root_path = Path(__file__).resolve().parent
        return root_path.joinpath("dexed_dataset.pickle")

    def _create_and_pickle(self):
        print("\nLoading and storing a fresh dataset....")
        self.dataset = DexedDataset((3.0, 1.0), 512, 256, 16000, midi_notes=((56, 75), ),
                                    data_augmentation=False, data_storage_root_path="/media/gwendal/Data/Datasets",
                                    check_constrains_consistency=False,
                                    vst_params_learned_as_categorical=self.params_learned_as_cat)
        with open(self._dataset_pickle_filepath, 'wb') as f:
            pickle.dump(self.dataset, f)

    def load(self) -> DexedDataset:
        if self.refresh_pickled_dataset:
            self._create_and_pickle()  # after this, we have a loaded dataset
        if self.dataset is not None:
            return self.dataset
        else:
            try:
                with open(self._dataset_pickle_filepath, 'rb') as f:
                    self.dataset = pickle.load(f)
                    print("\nLoaded dataset from pickle file.")
            except EnvironmentError:
                print("\nNo dataset available from '{}'".format(self._dataset_pickle_filepath))
                self.dataset = None
                self._create_and_pickle()
            return self.dataset


dataset_loader = TestDexedDatasetLoader(force_refresh=False, vst_params_learned_as_categorical='all<=32')


class SynthParamsLossTest(unittest.TestCase):
    def test_dexed_losses(self):
        ds = dataset_loader.load()
        idx_helper = ds.preset_indexes_helper
        total_cat_params = len(idx_helper.cat_idx_learned_as_cat) + len(idx_helper.cat_idx_learned_as_num)
        total_num_params = len(idx_helper.num_idx_learned_as_cat) + len(idx_helper.num_idx_learned_as_num)
        print("\nLearned params: {} numerical, {} categorical".format(total_num_params, total_cat_params))
        accuracy_criterion = model.loss.CategoricalParamsAccuracy(ds.preset_indexes_helper)
        # we test presets with all available algorithms
        nb_preset_with_algo = np.zeros((32,))
        preset_UIDs_and_algo = list()
        preset_idx = 0
        while np.any(nb_preset_with_algo < 1) and preset_idx < len(ds.valid_preset_UIDs):
            preset_UID = ds.valid_preset_UIDs[preset_idx]
            algo = ds.get_full_preset_params(preset_UID).get_full()[0, 4].item()
            algo = int(np.round(algo * 31.0))
            # Allow 2 presets max per algo
            if nb_preset_with_algo[algo] < 2:
                preset_UIDs_and_algo.append((preset_UID, algo))
                nb_preset_with_algo[algo] += 1
            preset_idx += 1
        for preset_UID, algo in preset_UIDs_and_algo:
            vst_preset_GT = ds.get_full_preset_params(preset_UID)
            learnable_preset_GT = vst_preset_GT.get_learnable()
            self.assertAlmostEqual(accuracy_criterion(learnable_preset_GT, learnable_preset_GT), 100.0, delta=0.1)
            # introduce an error on the preset, and measure the decreasing accuracy
            vst_preset_modified = copy.deepcopy(vst_preset_GT)  # we'll change the DX7 algo
            vst_preset_modified._full_presets[0, 4] = (vst_preset_modified._full_presets[0, 4] + 1.0/31) % 1.0
            output_preset = vst_preset_modified.get_learnable()
            acc = accuracy_criterion(output_preset, learnable_preset_GT)
            self.assertAlmostEqual(acc, 100.0 * (total_cat_params - 1.0) / total_cat_params, delta=0.1)
            # TODO test permutations of oscillators (depend on the algorithm)
            #    get the permutations from the Dexed class


if __name__ == '__main__':

    unittest.main()
