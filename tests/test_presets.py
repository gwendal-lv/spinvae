import unittest
from pathlib import Path
import pickle
import copy

import numpy as np
import torch
import torch.nn as nn

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


class LearnableParamsTest(unittest.TestCase):
    def test_dexed_params(self):
        ds = dataset_loader.load()
        idx_helper = ds.preset_indexes_helper
        for preset_UID in ds.valid_preset_UIDs:
            ds_preset = ds.get_full_preset_params(preset_UID)
            # test consistency between the VST preset and its learnable representation
            vst_preset = ds_preset.get_full()
            learn_preset = ds_preset.get_learnable()
            vst_idx_learned_as_num = {**idx_helper.num_idx_learned_as_num, **idx_helper.cat_idx_learned_as_num}
            for vst_index, learn_index in vst_idx_learned_as_num.items():  # Merged dict
                self.assertEqual(vst_preset[0, vst_index], learn_preset[0, learn_index])
            vst_idx_learned_as_cat = {**idx_helper.num_idx_learned_as_cat, **idx_helper.cat_idx_learned_as_cat}
            for vst_index, learn_indexes in vst_idx_learned_as_cat.items():
                vst_category_index = vst_preset[0, vst_index] * (idx_helper.vst_param_cardinals[vst_index] - 1.0)
                vst_category_index = int(np.round(vst_category_index.item()))
                one_hot_learn_category = learn_preset[0, learn_indexes]
                learn_category_index = torch.argmax(one_hot_learn_category).item()
                self.assertEqual(vst_category_index, learn_category_index)
                # also assert one-hot encoding (no indexing error)
                self.assertEqual(torch.count_nonzero(one_hot_learn_category).item(), 1)
            # TODO build a VST preset from its learnable representation, test consistency
            pass


def get_presets_for_all_algorithms(ds: DexedDataset):
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
    return preset_UIDs_and_algo


class LossTest(unittest.TestCase):
    def test_dexed_losses(self):
        ds = dataset_loader.load()
        idx_helper = ds.preset_indexes_helper
        total_cat_params = len(idx_helper.cat_idx_learned_as_cat) + len(idx_helper.cat_idx_learned_as_num)
        total_num_params = len(idx_helper.num_idx_learned_as_cat) + len(idx_helper.num_idx_learned_as_num)
        print("\nLearned params: {} VST-numerical, {} VST-categorical".format(total_num_params, total_cat_params))
        eval_criterion = model.loss.AccuracyAndQuantizedNumericalLoss(idx_helper, numerical_loss_type='L1')
        # TODO backprop loss
        # we test presets with all available algorithms
        # TODO build a batch of presets for testing
        preset_UIDs_and_algo = get_presets_for_all_algorithms(ds)
        learnable_preset_length = ds.get_full_preset_params(0).get_learnable().shape[1]
        learnable_preset_GT_batch = torch.empty((len(preset_UIDs_and_algo), learnable_preset_length))
        learnable_preset_num_err_batch = torch.empty((len(preset_UIDs_and_algo), learnable_preset_length))
        learnable_preset_cat_err_batch = torch.empty((len(preset_UIDs_and_algo), learnable_preset_length))
        for batch_index, (preset_UID, algo) in enumerate(preset_UIDs_and_algo):
            vst_preset_GT = ds.get_full_preset_params(preset_UID)
            learnable_preset_GT = vst_preset_GT.get_learnable()
            learnable_preset_GT_batch[batch_index, :] = learnable_preset_GT[0, :].clone()
            acc, num_error = eval_criterion(learnable_preset_GT, learnable_preset_GT)
            self.assertAlmostEqual(acc, 100.0, delta=0.1)
            self.assertAlmostEqual(num_error, 0.0, delta=1e-4)
            # - - - - - - - Test the metric losses (no backprop possible) - - - - - - -
            output_preset = None
            for vst_index in idx_helper.categorical_vst_params[0:6]:  # FIXME For each categorical VST param
                # introduce an error on the preset, and measure the decreasing accuracy
                if idx_helper.vst_param_learnable_model[vst_index] is not None:  # If the preset is learnable
                    vst_preset_modified = copy.deepcopy(vst_preset_GT)
                    card = idx_helper.vst_param_cardinals[vst_index]
                    cat_increment = 1.0 / (card - 1.0)
                    if np.isclose(vst_preset_modified._full_presets[0, vst_index].item(), 1.0):
                        vst_preset_modified._full_presets[0, vst_index] -= cat_increment
                    else:
                        vst_preset_modified._full_presets[0, vst_index] += cat_increment
                    output_preset = vst_preset_modified.get_learnable()
                    acc, num_error = eval_criterion(output_preset, learnable_preset_GT)
                    self.assertAlmostEqual(acc, 100.0 * (total_cat_params - 1.0) / total_cat_params, delta=0.1,
                                           msg="VST categorical param #{}, '{}'"
                                           .format(vst_index, idx_helper.vst_param_names[vst_index]))
            learnable_preset_cat_err_batch[batch_index, :] = output_preset[0, :].clone()
            for vst_index in idx_helper.numerical_vst_params[0:6]:  # FIXME For each numerical param
                if idx_helper.vst_param_learnable_model[vst_index] is not None:  # If the preset is learnable
                    vst_preset_modified = copy.deepcopy(vst_preset_GT)
                    if vst_preset_modified._full_presets[0, vst_index] < 0.5:
                        expected_loss = 1.0 - vst_preset_modified._full_presets[0, vst_index].item()
                        vst_preset_modified._full_presets[0, vst_index] = 1.0
                    else:
                        expected_loss = vst_preset_modified._full_presets[0, vst_index].item()
                        vst_preset_modified._full_presets[0, vst_index] = 0.0
                    expected_loss = expected_loss / total_num_params
                    output_preset = vst_preset_modified.get_learnable()
                    acc, num_error = eval_criterion(output_preset, learnable_preset_GT)  # Quantized L1 loss
                    self.assertAlmostEqual(expected_loss, num_error, delta=1e-4)
            learnable_preset_num_err_batch[batch_index, :] = output_preset[0, :].clone()
            # TODO - - - - - - - test permutations of oscillators (depend on the algorithm) - - - - - - -
            #    get the permutations from the Dexed class
            # TODO test minimal loss for several permutations
        # TODO test all data as a batch
        acc, num_error = eval_criterion(learnable_preset_GT_batch, learnable_preset_GT_batch)
        self.assertAlmostEqual(acc, 100.0, delta=0.1)
        self.assertAlmostEqual(num_error, 0.0, delta=1e-4)
        acc, num_error = eval_criterion(learnable_preset_cat_err_batch, learnable_preset_GT_batch, bkpt=True)
        self.assertAlmostEqual(acc, 100.0 * (total_cat_params - 1.0) / total_cat_params, delta=0.1)
        acc, num_error = eval_criterion(learnable_preset_num_err_batch, learnable_preset_GT_batch)
        self.assertAlmostEqual(acc, 100.0, delta=0.1)


if __name__ == '__main__':

    unittest.main()
