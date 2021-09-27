"""
A Dataset that merges different datasets (e.g. Dexed, Surge, NSynth, ...).
Can be used for pre-training a part of a neural network (e.g. the audio VAE only, without preset inference)
"""

import numpy as np
from typing import List
from collections import OrderedDict

import torch

from data.abstractbasedataset import AudioDataset
from data.nsynthdataset import NsynthDataset
from data.surgedataset import SurgeDataset
from synth import surge
from data.dexeddataset import DexedDataset

from data import sampler



class MergedDataset(AudioDataset):
    def __init__(self, model_config, dataset_type: str):
        """

        Subsets of data:
            - NSynth: 'train' and 'valid_test_merged' sets cannot be modified. No 'test' set.
            - Surge: 'train' and 'validation' will be split 80% / 20%. No 'test' set.
            - Dexed: 'train' and 'validation' sets use the special -1 k-fold (the will share data with the usual 5-fold
              train/valid sets). 'test' set will be held out

        This class builds subsets of data for each synth, then merges the results to provide torch sub-samplers.

        :param model_config:
        :param dataset_type: 'full', 'train' or 'validation'
        """
        from data import dataset  # local import to prevent recursive import issues
        super().__init__(** dataset.model_config_to_dataset_kwargs(model_config))

        self.dataset_type = dataset_type
        if self.dataset_type not in ['full', 'train', 'validation' ]:
            raise ValueError("Invalid dataset type '{}'".format(self.dataset_type))
        if self.dataset_type in ['full', 'train']:
            _nsynth_dataset_type = self.dataset_type
            self._data_augmentation = True
        else:
            _nsynth_dataset_type = 'valid_test_merged'
            self._data_augmentation = False
        self._datasets = {'NSynth': NsynthDataset(** dataset.model_config_to_dataset_kwargs(model_config),
                                                  dataset_type=_nsynth_dataset_type,
                                                  exclude_instruments_with_missing_notes=True,
                                                  exclude_sonic_qualities=['reverb'],
                                                  force_include_all_acoustic=True,
                                                  data_augmentation=self._data_augmentation,
                                                  random_seed=self._random_seed),
                          'Surge': SurgeDataset(** dataset.model_config_to_dataset_kwargs(model_config),
                                                fx_bypass_level=surge.FxBypassLevel.ALL,
                                                data_augmentation=self._data_augmentation),
                          'Dexed': DexedDataset(** dataset.model_config_to_dataset_kwargs(model_config),
                                                data_augmentation=self._data_augmentation)
                          }
        # Dataset items that belong to the current config (full/train/valid)
        self._local_indexes = {'NSynth': np.arange(len(self._datasets['NSynth']))}  # NSynth already configured
        if self.dataset_type == 'full':
            for synth_name in ['Surge', 'Dexed']:
                self._local_indexes[synth_name] = np.arange(len(self._datasets[synth_name]))
        else:
            # retrieve Surge and Dexed indexes - no 'test' set for Surge (will be used for pretrain only)
            dexed_split_indexes = sampler.get_subsets_indexes(self._datasets['Dexed'], k_fold=-1,
                                                              random_seed=self._random_seed)
            surge_split_indexes = sampler.get_subsets_indexes(self._datasets['Surge'], k_fold=0, k_folds_count=5,
                                                              test_holdout_proportion=0.0, random_seed=self._random_seed)
            self._local_indexes['Surge'] = np.sort(surge_split_indexes[self.dataset_type])
            self._local_indexes['Dexed'] = np.sort(dexed_split_indexes[self.dataset_type])

        # Conversions: 'local' UIDs (in their original datasets) vs. 'global' UID in this merged dataset class
        self._global_UID_offsets = OrderedDict()  # To ensure iteration in increasing order
        self._global_UID_offsets["Dexed"] = 0  # max Dexed UID: 393878 (a lot of original presets were duplicates)
        self._global_UID_offsets["Surge"] = 400000  # 2300 preset (so we leave 100000 slots empty)
        self._global_UID_offsets["NSynth"] = 500000

        # TODO generate available presets' global UIDs
        self.valid_preset_UIDs = list()
        for synth_name, ds in self._datasets.items():
            for local_index in self._local_indexes[synth_name]:
                self.valid_preset_UIDs.append(ds.valid_preset_UIDs[local_index] + self._global_UID_offsets[synth_name])
        self.valid_preset_UIDs = np.sort(self.valid_preset_UIDs)

        i = 0

    def _global_to_local_UID_and_ds(self, global_UID):
        """ Converts a global UID (e.g. can be > 500 000) into a (local_UID, local_dataset) tuple. """
        for synth_name, UID_offset in reversed(self._global_UID_offsets.items()):  # ordered, increasing UID offsets
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

    def get_name_from_preset_UID(self, preset_UID: int) -> str:
        local_UID, ds = self._global_to_local_UID_and_ds(preset_UID)
        return "[{}] {}".format(ds.synth_name[:3], ds.get_name_from_preset_UID(local_UID))

    def get_wav_file(self, preset_UID, midi_note, midi_velocity, variation=0):
        local_UID, ds = self._global_to_local_UID_and_ds(preset_UID)
        return ds.get_wav_file(local_UID, midi_note, midi_velocity, variation)

    def get_audio_file_stem(self, preset_UID, midi_note, midi_velocity, variation=0):
        local_UID, ds = self._global_to_local_UID_and_ds(preset_UID)
        return ds.get_audio_file_stem(local_UID, midi_note, midi_velocity, variation)

    # ============================ Overridden Audio Dataset methods ==========================

    # TODO GET ITEM OVERRIDE

    def get_nb_variations_per_note(self, preset_UID=-1):
        local_UID, ds = self._global_to_local_UID_and_ds(preset_UID)
        return ds.get_nb_variations_per_note(local_UID)

    def get_spec_file_path(self, preset_UID, midi_note, midi_velocity, variation=0):
        local_UID, ds = self._global_to_local_UID_and_ds(preset_UID)
        return ds.get_spec_file_path(local_UID,midi_note, midi_velocity, variation)

    def _load_spectrogram_stats(self):
        raise NotImplementedError()  # TODO load stats from each merged dataset, extract global stats

    @property
    def excluded_patches_UIDs(self):
        raise NotImplementedError()  # TODO read excluded patches from all child classes, convert UIDs and return

    # =============== Disabled Audio Dataset methods (we don't write any audio files from this class) =============

    def compute_and_store_spectrograms_and_stats(self):
        raise AssertionError("This merged dataset instance cannot generate audio files or pre-compute spectrograms. "
                             "Please generates those files from the original datasets first.")

    @property
    def _spectrograms_folder(self):
        raise AssertionError("This class handles multiple spectrograms folders.")
