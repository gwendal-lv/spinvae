import unittest
from pathlib import Path
import pickle
import copy

import numpy as np
import torch
import torch.nn as nn

from data.dataset import DexedDataset


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
    def test_dexed_permutations(self):
        from synth import dexedpermutations  # This module uses >= 1 indices (consistency with DX7 manual)
        found_algos = [False for _ in range(32)]
        for feedback, permutations_dict \
            in {'with_feedback': dexedpermutations._osc_permutations_per_algo_with_feedback,
                'without_feedback': dexedpermutations._osc_permutations_per_algo_without_feedback}.items():
            for algo, permutations in permutations_dict.items():
                found_algos[algo-1] = True
                for i in range(0, permutations.shape[0]):  # Check that all permutations are different
                    perm = permutations[i, :]
                    for j in range(i+1, permutations.shape[0]):
                        self.assertTrue(np.any(perm != permutations[j, :]),
                                        'duplicated permutation (algo {}, {})'.format(algo, feedback))
                    # Also check that permutation contains all elements
                    self.assertGreaterEqual(perm.min(), 1)
                    self.assertLessEqual(perm.max(), 6)
                    self.assertEqual(len(set(perm)), 6)
            self.assertTrue(all(found_algos))

    def test_dexed_params(self):
        ds = dataset_loader.load()
        idx_helper = ds.preset_indexes_helper

        learn_preset = None
        for preset_UID in ds.valid_preset_UIDs[0:10]:
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

        # dev tests
        batch_size = 10
        learn_presets_batch = torch.empty((batch_size, learn_preset.shape[1]))
        for i, preset_UID in enumerate(ds.valid_preset_UIDs[0:batch_size]):
            learn_presets_batch[i, :] = ds.get_full_preset_params(preset_UID).get_learnable()
        idx_helper.get_symmetrical_learnable_presets(learn_presets_batch, learn_presets_batch)
        # TODO test permutations of oscillators (check consistency after permutation)


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
        raise NotImplementedError("This test is deprecated, was based on obsolete synth param losses")
        #eval_criterion = model.loss.AccuracyAndQuantizedNumericalLoss(idx_helper, numerical_loss_type='L1',
        #                                                              compute_symmetrical_presets=True)
        #backprop_criterion = model.loss.SynthParamsLoss(idx_helper, normalize_losses=True, cat_softmax=True,
        #                                                compute_symmetrical_presets=True)
        # TODO backprop loss
        # we test presets with all available algorithms
        preset_UIDs_and_algo = get_presets_for_all_algorithms(ds)
        learnable_preset_length = ds.get_full_preset_params(0).get_learnable().shape[1]
        # build batches of presets (see end of this test)
        learnable_preset_GT_batch = torch.empty((len(preset_UIDs_and_algo), learnable_preset_length))
        learnable_preset_num_err_batch = torch.empty((len(preset_UIDs_and_algo), learnable_preset_length))
        learnable_preset_cat_err_batch = torch.empty((len(preset_UIDs_and_algo), learnable_preset_length))
        learnable_preset_symmetry_batch = torch.empty((len(preset_UIDs_and_algo), learnable_preset_length))
        for batch_index, (preset_UID, algo) in enumerate(preset_UIDs_and_algo):
            vst_preset_GT = ds.get_full_preset_params(preset_UID)
            learnable_preset_GT = vst_preset_GT.get_learnable()
            learnable_preset_GT_batch[batch_index, :] = learnable_preset_GT[0, :].clone()
            acc, num_error = eval_criterion(learnable_preset_GT, learnable_preset_GT)
            self.assertAlmostEqual(acc, 100.0, delta=0.1)
            self.assertAlmostEqual(num_error, 0.0, delta=1e-4)
            # backprop_loss = backprop_criterion(learnable_preset_GT, learnable_preset_GT)  # > 0
            # self.assertAlmostEqual(backprop_loss, 0.0, delta=1e-3)
            # - - - - - - - Test numerical loss and categorical accuracy - - - - - - -
            output_preset = None
            for vst_index in idx_helper.categorical_vst_params[0:6]:  # FIXME For each categorical VST param
                # introduce an error on the preset, and measure the decreasing accuracy
                if idx_helper.vst_param_learnable_model[vst_index] is not None:  # If the preset is learnable
                    vst_preset_modified = copy.deepcopy(vst_preset_GT)
                    if vst_index != 4:  # all but algo
                        card = idx_helper.vst_param_cardinals[vst_index]
                        cat_increment = 1.0 / (card - 1.0)
                        if np.isclose(vst_preset_modified._full_presets[0, vst_index].item(), 1.0):
                            vst_preset_modified._full_presets[0, vst_index] -= cat_increment
                        else:
                            vst_preset_modified._full_presets[0, vst_index] += cat_increment
                    else:  # Algo has a very different value, because many algo are symmetrical without feedback
                        if vst_preset_modified._full_presets[0, vst_index].item() < 0.5:
                            vst_preset_modified._full_presets[0, vst_index] = 1.0
                        else:
                            vst_preset_modified._full_presets[0, vst_index] = 0.0
                    output_preset = vst_preset_modified.get_learnable()
                    acc, num_error = eval_criterion(output_preset, learnable_preset_GT)
                    # Be careful about this test: the "permutations" loss might give a (truly) better acc than expected
                    self.assertAlmostEqual(acc, 100.0 * (total_cat_params - 1.0) / total_cat_params, delta=0.1,
                                           msg="VST categorical param #{}, '{}'"
                                           .format(vst_index, idx_helper.vst_param_names[vst_index]))
            learnable_preset_cat_err_batch[batch_index, :] = output_preset[0, :].clone()
            for vst_index in idx_helper.numerical_vst_params[0:6]:  # FIXME For each numerical param (LONGER++)
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
            # - - - - - - - test permutations of oscillators (depend on the algorithm) - - - - - - -
            vst_preset_modified = copy.deepcopy(vst_preset_GT)
            no_feedback = np.isclose(vst_preset_GT._full_presets[0, 5].item(), 0)
            if algo == 2 and no_feedback:  # algo 3 --> algo 4 without feedback
                vst_preset_modified._full_presets[0, 4] = 3 / 31  # OK, another algo triggers test failure
            if algo == 5 and no_feedback:  # algo 6 --> algo 5 without feedback
                vst_preset_modified._full_presets[0, 4] = 4 / 31  # OK, another algo triggers test failure
            elif algo in [4, 5]:  # Algo 5 and 6: swap osc 1-2 and 3-4
                osc_12_backup = vst_preset_modified._full_presets[0, 23:67].clone()
                vst_preset_modified._full_presets[0, 23:67] = vst_preset_modified._full_presets[0, 67:111].clone()
                vst_preset_modified._full_presets[0, 67:111] = osc_12_backup
            elif algo in [11, 12, 18, 20, 21, 22]:  # Algo 12, 13, 19, 21, 22, 23, (24, 25): swap osc 4 and 5
                osc_4_backup = vst_preset_modified._full_presets[0, 89:111].clone()
                vst_preset_modified._full_presets[0, 89:111] = vst_preset_modified._full_presets[0, 111:133].clone()
                vst_preset_modified._full_presets[0, 111:133] = osc_4_backup
            elif algo in [23, 24, 28, 29, 30, 31]:  # DX7 Algo 24, 25, 29, 30, 31, 32: swap osc 1 and 2
                osc_1_backup = vst_preset_modified._full_presets[0, 23:45].clone()
                vst_preset_modified._full_presets[0, 23:45] = vst_preset_modified._full_presets[0, 45:67].clone()
                vst_preset_modified._full_presets[0, 45:67] = osc_1_backup
            else:  # Default, no symmetry applied....
                pass
            output_preset = vst_preset_modified.get_learnable()
            acc, num_error = eval_criterion(output_preset, learnable_preset_GT)  # Should remain 0 with symmetry
            self.assertAlmostEqual(acc, 100.0, delta=0.1)
            self.assertAlmostEqual(num_error, 0.0, delta=1e-4)
            learnable_preset_symmetry_batch[batch_index, :] = output_preset[0, :].clone()
        # test all data as a batch
        acc, num_error = eval_criterion(learnable_preset_GT_batch, learnable_preset_GT_batch)  # GT vs GT
        self.assertAlmostEqual(acc, 100.0, delta=0.1)
        self.assertAlmostEqual(num_error, 0.0, delta=1e-4)
        backprop_loss = backprop_criterion(learnable_preset_GT_batch, learnable_preset_GT_batch)
        self.assertAlmostEqual(backprop_loss, 0.0, delta=1e-3)
        acc, num_error = eval_criterion(learnable_preset_symmetry_batch, learnable_preset_GT_batch)  # Symmetry vs GT
        self.assertAlmostEqual(acc, 100.0, delta=0.1)
        self.assertAlmostEqual(num_error, 0.0, delta=1e-4)
        backprop_loss = backprop_criterion(learnable_preset_symmetry_batch, learnable_preset_GT_batch)
        self.assertAlmostEqual(backprop_loss, 0.0, delta=1e-3)
        acc, num_error = eval_criterion(learnable_preset_cat_err_batch, learnable_preset_GT_batch)  # cat err
        self.assertAlmostEqual(acc, 100.0 * (total_cat_params - 1.0) / total_cat_params, delta=0.1)
        self.assertAlmostEqual(num_error, 0.0, delta=1e-4)
        backprop_loss = backprop_criterion(learnable_preset_cat_err_batch, learnable_preset_GT_batch)
        self.assertGreater(backprop_loss, 1e-2)
        acc, num_error = eval_criterion(learnable_preset_num_err_batch, learnable_preset_GT_batch)  # num err
        self.assertAlmostEqual(acc, 100.0, delta=0.1)
        self.assertGreater(num_error, 0.0)
        backprop_loss = backprop_criterion(learnable_preset_num_err_batch, learnable_preset_GT_batch)
        self.assertGreater(backprop_loss, 1e-3)



if __name__ == '__main__':

    unittest.main()
