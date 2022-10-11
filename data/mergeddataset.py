"""
A Dataset that merges different datasets (e.g. Dexed, Surge, NSynth, ...).
Can be used for pre-training a part of a neural network (e.g. the audio VAE only, without preset inference)
"""
import warnings

import numpy as np
from typing import List, Tuple, Dict
from collections import OrderedDict

import torch
import torch.utils.data

from data.abstractbasedataset import AudioDataset
from data.nsynthdataset import NsynthDataset
from data.surgedataset import SurgeDataset
from synth import surge
from data.dexeddataset import DexedDataset

from data import sampler



class MergedDataset(AudioDataset):
    def __init__(self, model_config, dataset_type: str, dummy_synth_params_tensor=False,
                 k_folds_count=5, test_holdout_proportion=0.2):
        """

        Subsets of data:
            - NSynth: 'train' and 'valid_test_merged' sets cannot be modified. No 'test' set.
            - Surge: 'train' and 'validation' will be split 80% / 20%. No 'test' set.
            - Dexed: 'train' and 'validation' sets use the special -1 k-fold (the will share data with the usual 5-fold
              train/valid sets). 'test' set will be held out

        This class builds subsets of data for each synth, then merges the results to provide torch sub-samplers.

        :param model_config:
        :param dataset_type: 'full', 'train' or 'validation'
        :param dummy_synth_params_tensor: If True, a 0 single-element tensor will be returned by the __getitem__
            method (for compatibility with a PresetDataset)
        """
        from data import dataset  # local import to prevent recursive import issues
        super().__init__(** dataset.model_config_to_dataset_kwargs(model_config))

        self.dummy_synth_params_tensor = dummy_synth_params_tensor
        self.dataset_type = dataset_type
        if self.dataset_type not in ['full', 'train', 'validation' ]:
            raise ValueError("Invalid dataset type '{}'".format(self.dataset_type))
        if self.dataset_type in ['full', 'train']:
            _nsynth_dataset_type = self.dataset_type
            self._data_augmentation = True
        else:
            _nsynth_dataset_type = 'valid_test_merged'
            self._data_augmentation = False
        self._datasets = {'Dexed': DexedDataset(** dataset.model_config_to_dataset_kwargs(model_config),
                                                data_augmentation=self._data_augmentation),
                          'Surge': SurgeDataset(** dataset.model_config_to_dataset_kwargs(model_config),
                                                fx_bypass_level=surge.FxBypassLevel.ALL,
                                                data_augmentation=self._data_augmentation),
                          'NSynth': NsynthDataset(** dataset.model_config_to_dataset_kwargs(model_config),
                                                  dataset_type=_nsynth_dataset_type,
                                                  exclude_instruments_with_missing_notes=True,
                                                  exclude_sonic_qualities=['reverb'],
                                                  force_include_all_acoustic=True,
                                                  data_augmentation=self._data_augmentation,
                                                  random_seed=self._random_seed,
                                                  required_midi_notes=model_config.required_dataset_midi_notes),
                          }
        # Dataset items that belong to the current config (full/train/valid)
        self._local_indexes = {'NSynth': np.arange(len(self._datasets['NSynth']))}  # NSynth already configured
        if self.dataset_type == 'full':
            for synth_name in ['Surge', 'Dexed']:
                self._local_indexes[synth_name] = np.arange(len(self._datasets[synth_name]))
        else:
            # retrieve Surge and Dexed indexes - no 'test' set for Surge (will be used for pretrain only)
            dexed_split_indexes = sampler.get_subsets_indexes(
                self._datasets['Dexed'], k_fold=-1, k_folds_count=k_folds_count,
                test_holdout_proportion=test_holdout_proportion, random_seed=self._random_seed)
            surge_split_indexes = sampler.get_subsets_indexes(
                self._datasets['Surge'], k_fold=-1, k_folds_count=k_folds_count,
                test_holdout_proportion=0.0, random_seed=self._random_seed)
            self._local_indexes['Surge'] = np.sort(surge_split_indexes[self.dataset_type])
            self._local_indexes['Dexed'] = np.sort(dexed_split_indexes[self.dataset_type])

        # Conversions: 'local' UIDs (in their original datasets) vs. 'global' UID in this merged dataset class, also
        # for indices. OrderedDict of both, to ensure order when iterating over dict keys
        self._global_UID_offsets = OrderedDict()
        self._global_UID_offsets["Dexed"] = 0  # max Dexed UID: 393878 (a lot of original presets were duplicates)
        self._global_UID_offsets["Surge"] = 400000  # 2300 preset (so we leave 100000 slots empty)
        self._global_UID_offsets["NSynth"] = 500000
        # generate available presets' global UIDs
        self.valid_preset_UIDs = list()
        for synth_name, ds in self._datasets.items():
            for local_index in self._local_indexes[synth_name]:  # Here, we virtually create sub-datasets
                self.valid_preset_UIDs.append(ds.valid_preset_UIDs[local_index] + self._global_UID_offsets[synth_name])
        self.valid_preset_UIDs = np.sort(self.valid_preset_UIDs)
        # Offset: global indexes (to know to which synth an index of this class belongs)
        self._global_indices_offset = OrderedDict()
        next_index = 0
        for synth_name in self._global_UID_offsets.keys():
            self._global_indices_offset[synth_name] = next_index
            next_index += len(self._local_indexes[synth_name])

    def __getitem__(self, i):
        if not self.dummy_synth_params_tensor:
            return super().__getitem__(i)
        else:
            spec, uid, notes, labels = super().__getitem__(i)
            return spec, torch.zeros((1, )), uid, notes, labels

    def _global_to_local_UID_and_ds(self, global_UID: int) -> Tuple[int, AudioDataset]:
        """ Converts a global UID (e.g. can be > 500 000) into a (local_UID, local_dataset) tuple. """
        for synth_name, UID_offset in reversed(self._global_UID_offsets.items()):  # ordered, rev-increasing UID offsets
            if global_UID >= UID_offset:
                return global_UID - UID_offset, self._datasets[synth_name]
        raise ValueError("Could not convert the given global UID into a local UID")

    # ========================= Abstract methods (required for any Audio Dataset) =======================

    @property
    def synth_name(self):
        return "Merged_Dexed-Surge-NSynth"

    @property
    def total_nb_presets(self):  # Sum of available indexes for each dataset
        return sum([ds.total_nb_presets for _, ds in self._datasets.items()])

    def get_name_from_preset_UID(self, preset_UID: int, long_name=False) -> str:
        local_UID, ds = self._global_to_local_UID_and_ds(preset_UID)
        return "[{}] {}".format(ds.synth_name[:3], ds.get_name_from_preset_UID(local_UID))

    def get_wav_file(self, preset_UID, midi_note, midi_velocity, variation=0):
        local_UID, ds = self._global_to_local_UID_and_ds(preset_UID)
        return ds.get_wav_file(local_UID, midi_note, midi_velocity, variation)

    def get_audio_file_stem(self, preset_UID, midi_note, midi_velocity, variation=0):
        local_UID, ds = self._global_to_local_UID_and_ds(preset_UID)
        return ds.get_audio_file_stem(local_UID, midi_note, midi_velocity, variation)

    # ============================ Overridden Audio Dataset methods ==========================
    # __getitem__ does not need to be overridden: it works with valid_preset_UIDs (which refer to sub-datasets only)

    def get_nb_variations_per_note(self, preset_UID=-1):
        local_UID, ds = self._global_to_local_UID_and_ds(preset_UID)
        return ds.get_nb_variations_per_note(local_UID)

    def get_spec_file_path(self, preset_UID, midi_note, midi_velocity, variation=0):
        local_UID, ds = self._global_to_local_UID_and_ds(preset_UID)
        return ds.get_spec_file_path(local_UID, midi_note, midi_velocity, variation)

    def _load_spectrogram_stats(self):
        raise NotImplementedError()  # TODO load stats from each merged dataset, extract global stats

    @property
    def excluded_patches_UIDs(self):
        raise NotImplementedError()  # TODO read excluded patches from all child classes, convert UIDs and return

    @property
    def _available_labels_path(self):
        raise AssertionError("A mergeddataset does not have a single _available_labels_path. Please get this property "
                             "from the sub-datasets.")

    @property
    def available_labels_names(self):
        # TODO get labels names from all sub-datasets, check that they are identical
        prev_labels_names = None
        for ds_name, ds in self._datasets.items():
            labels_names = ds.available_labels_names
            if prev_labels_names is not None and prev_labels_names != labels_names:
                raise ValueError("Some sub-datasets have different labels. Current dataset: {}, labels={}. "
                                 "Previously found labels: {}".format(ds_name, labels_names, prev_labels_names))
            prev_labels_names = labels_names
        return labels_names

    def get_labels_tensor(self, preset_UID):
        local_UID, ds = self._global_to_local_UID_and_ds(preset_UID)
        return ds.get_labels_tensor(local_UID)

    def get_labels_name(self, preset_UID: int) -> List[str]:
        local_UID, ds = self._global_to_local_UID_and_ds(preset_UID)
        return ds.get_labels_name(local_UID)

    def get_original_instrument_family(self, preset_UID: int) -> str:
        local_UID, ds = self._global_to_local_UID_and_ds(preset_UID)
        return ds.get_original_instrument_family(local_UID)

    def save_labels(self, labels_names: List[str], labels_per_UID: Dict[int, List[str]]):
        raise NotImplementedError("Labels must be extract for each sub-dataset separately.")

    # =============== Disabled Audio Dataset methods (we don't write any audio files from this class) =============

    def compute_and_store_spectrograms_and_stats(self):
        raise AssertionError("This merged dataset instance cannot generate audio files or pre-compute spectrograms. "
                             "Please generates those files from the original datasets first.")

    @property
    def _spectrograms_folder(self):
        raise AssertionError("This class handles multiple spectrograms folders.")

    # ============================ Dataloaders with Weighted Samplers ==========================

    def get_dataloader(self, batch_size: int,
                       use_weighted_sampler=False, max_imbalance_ratio=10.0,
                       num_workers=0, persistent_workers=True, pin_memory=False):
        """ Returns a dataloader properly configured to access this dataset, and the nb of items returns by this
        dataloader for each epoch.

        :param batch_size: (PyTorch DataLoader arg)
        :param use_weighted_sampler: If True, the number of items from overrepresented synths available at each
           will be reduced using a pytorch WeightedRandomSampler.
        :param max_imbalance_ratio: The max ratio between the number of items of an overrepresented synth and
           and underrepresented synth or dataset (e.g. NSynth has very few instruments) (used if use_weighted_sampler
           is True). This is not a class imbalance ratio, but a sub-dataset imbalance ratio.
        :param num_workers: (PyTorch DataLoader arg)
        :param persistent_workers: (PyTorch DataLoader arg)
        :param pin_memory: (PyTorch DataLoader arg)
        """
        drop_last = (self.dataset_type == 'train')
        if self.dataset_type == 'validation' and use_weighted_sampler:
            warnings.warn("The validation dataloader will use a weighted random sampler, which should be avoided to "
                          "ensure consistent validation results.")
        torch_rng = torch.Generator().manual_seed(sampler._SEED_OFFSET + self._random_seed)
        if not use_weighted_sampler:
            random_sampler = torch.utils.data.RandomSampler(self, generator=torch_rng)
        else:
            # Find the smallest 'reference' dataset (ratios computed from this one's length)
            ds_len = {name: len(indices) for name, indices in self._local_indexes.items()}
            # smallest_ds_name = min(ds_lengths, key=ds_lengths.get)
            smallest_ds_len = min([len(indices) for _, indices in self._local_indexes.items()])
            # For each synth, we set the proper weight for each global index
            weights = np.ones((len(self), )) * -1.0
            ds_len_after = dict()  # target lengths, after 'subsetting' using the weighted sampler
            for synth_name, index_offset in self._global_indices_offset.items():  # Ordered dict (increasing offsets)
                if ds_len[synth_name] < max_imbalance_ratio * smallest_ds_len:
                    ds_len_after[synth_name] = ds_len[synth_name]
                    w = 1.0
                else:
                    ds_len_after[synth_name] = int(max_imbalance_ratio * smallest_ds_len)
                    w = ds_len_after[synth_name] / ds_len[synth_name]
                for i in range(len(self._local_indexes[synth_name])):
                    weights[i + index_offset] = w
            num_samples_after = sum([l for _, l in ds_len_after.items()])
            random_sampler = torch.utils.data.WeightedRandomSampler(weights=weights, num_samples=num_samples_after,
                                                                    generator=torch_rng)
        # Actual nb of items depends on: wsampler or not, and drop_last or not
        if not use_weighted_sampler:
            if drop_last:
                nb_dataloader_items = (len(self) // batch_size) * batch_size
            else:
                nb_dataloader_items = len(self)
        else:
            if drop_last:
                nb_dataloader_items = (len(random_sampler) // batch_size) * batch_size
            else:
                nb_dataloader_items = len(random_sampler)
        return (torch.utils.data.DataLoader(self, batch_size=batch_size, sampler=random_sampler, num_workers=num_workers,
                                            pin_memory=pin_memory, drop_last=drop_last,
                                            persistent_workers=((num_workers > 0) and persistent_workers)),
                nb_dataloader_items)


