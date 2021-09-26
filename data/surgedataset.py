"""
This file allows to render audio file and use the generated Surge dataset.
Data is formatted such that it is compatible with other datasets (Dexed synth, NSynth audio, ...).
However, this dataset does not provide Surge synthesis parameters.
"""
import multiprocessing
import pathlib
import json
import copy
import shutil
import warnings
from datetime import datetime
import os
from typing import Optional

import numpy as np
import torchaudio
import soundfile
from natsort import natsorted

from data import abstractbasedataset
from synth import surge


class SurgeDataset(abstractbasedataset.AudioDataset):
    def __init__(self, note_duration, n_fft, fft_hop, Fs,
                 midi_notes=((60, 100),), multichannel_stacked_spectrograms=False,
                 n_mel_bins=-1, mel_fmin=0, mel_fmax=8000,
                 normalize_audio=False, spectrogram_min_dB=-120.0,
                 spectrogram_normalization: Optional[str] = 'min_max',
                 data_storage_root_path: Optional[str] = None,
                 random_seed=0, data_augmentation=True,
                 fx_bypass_level=surge.FxBypassLevel.ALL,
                 check_consistency=True):
        """
        Class for rendering an audio dataset for the Surge synth. Can be used by a PyTorch DataLoader.

        Please refer to abstractbasedataset.AudioDataset for documentation about constructor arguments.

        :param fx_bypass_level: Describes how much Surge FX should be bypassed.
        :param check_consistency: Checks if generated data used the same arguments (e.g. same FX disabled, ...)
            as this constructor
        """
        super().__init__(note_duration, n_fft, fft_hop, Fs, midi_notes, multichannel_stacked_spectrograms, n_mel_bins,
                         mel_fmin, mel_fmax, normalize_audio, spectrogram_min_dB, spectrogram_normalization,
                         data_storage_root_path, random_seed, data_augmentation)
        self._synth = surge.Surge(reduced_Fs=Fs, midi_note_duration_s=note_duration[0],
                                  render_duration_s=note_duration[0]+note_duration[1],
                                  fx_bypass_level=fx_bypass_level)  # FIXME
        # All available presets are considered valid
        self.valid_preset_UIDs = [self._synth.get_patch_info(idx)['UID'] for idx in range(self._synth.n_patches)]
        for UID in self.excluded_patches_UIDs:
            # Trying to exclude a preset that does not exist is considered a fatal error (please triple-check the list
            # of excluded UIDs). So, we don't try to catch a potential ValueError here
            self.valid_preset_UIDs.remove(UID)

        # Final init and checks
        if check_consistency:
            self._check_consistency()

    @property
    def synth_name(self):
        return "Surge"

    @property
    def total_nb_presets(self):
        return self._synth.n_patches

    def get_patch_info(self, preset_UID: int):
        patch_index = self._synth.find_index_from_UID(preset_UID)
        return self._synth.get_patch_info(patch_index)

    def get_name_from_preset_UID(self, preset_UID: int) -> str:
        return self.get_patch_info(preset_UID)['patch_name']

    @property
    def nb_variations_per_note(self):
        return self._synth.nb_variations_per_note

    def _check_consistency(self):
        with open(self._get_patches_info_path(), 'r') as f:
            d_info = json.load(f)
            if not self._synth.check_description(d_info['surge']):
                raise ValueError("Incoherent Surge arguments in {} (current: {}, found: {})"
                                 .format(self._get_patches_info_path(), self._synth.dict_description, d_info['surge']))

    def get_wav_file(self, preset_UID, midi_note, midi_velocity, variation=0):
        file_path = self._get_wav_file_path(preset_UID, midi_note, midi_velocity, variation)
        try:
            return soundfile.read(file_path)
        except RuntimeError:
            raise RuntimeError("Can't open file {}. Please pre-render audio files for this "
                               "dataset configuration.".format(file_path))

    def _get_patches_info_path(self):
        return self.data_storage_path.joinpath("dataset_info.json")

    def get_audio_file_stem(self, preset_UID, midi_note, midi_velocity, variation=0):
        return "{:04d}_pitch{:03d}vel{:03d}_var{:03d}".format(preset_UID, midi_note, midi_velocity, variation)

    def _get_wav_file_path(self, patch_UID, midi_pitch, midi_vel, variation):
        return self.data_storage_path\
            .joinpath("Audio/{}.wav".format(self.get_audio_file_stem(patch_UID, midi_pitch, midi_vel, variation)))

    @property
    def _normalized_audio_file_path(self):
        return self.data_storage_path.joinpath("normalized_amplitudes.txt")

    def generate_wav_files(self):
        print("Surge audio files rendering...")
        t_start = datetime.now()
        if os.path.exists(self.data_storage_path.joinpath("Audio")):
            shutil.rmtree(self.data_storage_path.joinpath("Audio"))
        self.data_storage_path.joinpath("Audio").mkdir(parents=False, exist_ok=False)
        # 0) Clean previous data
        open(self._normalized_audio_file_path, 'w').close()
        # 1) retrieve data for the .json file and for audio rendering
        dataset_info = {'surge': self._synth.dict_description,
                        'midi_notes': {'description': '(pitch, velocity) available for all patches',
                                       'notes': self.midi_notes},
                        'data_augmentation': {'description': 'Number of variations available for each note',
                                              'nb_variations': self.nb_variations_per_note},
                        'patches_count': self.valid_presets_count,
                        'patches': list()}
        valid_preset_indexes = list()
        for idx in range(self.total_nb_presets):
            p_info = self._synth.get_patch_info(idx)
            if p_info['UID'] in self.valid_preset_UIDs:  # We generate valid UIDs only
                dataset_info['patches'].append(p_info)
                valid_preset_indexes.append(idx)
        # 2) multi-processed audio rendering
        num_workers = os.cpu_count()
        split_preset_indexes = np.array_split(valid_preset_indexes, num_workers)
        with multiprocessing.Pool(num_workers) as p:  # automatically closes and joins all workers
            p.map(self._generate_wav_files_batch, split_preset_indexes)
        # 3) Write results
        with open(self._get_patches_info_path(), 'w') as f:
            json.dump(dataset_info, f, indent=4)
        delta_t = (datetime.now() - t_start).total_seconds()
        print("Finished writing {} .wav files ({:.1f}min total, {:.1f}ms/file using {} CPUs)"
              .format(self.nb_valid_audio_files, delta_t/60.0, 1000.0*delta_t/self.nb_valid_audio_files, num_workers))
        # 4) Sort file of forced-normalizations
        with open(self._normalized_audio_file_path, 'r') as f:
            lines = f.readlines()
        with open(self._normalized_audio_file_path, 'w') as f:
            f.writelines(natsorted(lines))

    def _generate_wav_files_batch(self, preset_indexes):
        Fs = -1
        for idx in preset_indexes:
            # pre-render all notes before volume reduction (if required) and writing files
            audio_renders = list()
            max_amplitude = -1.0
            for midi_note in self.midi_notes:
                for variation in range(self.nb_variations_per_note):
                    audio, Fs = self._synth.render_note(idx, midi_note[0], midi_note[1], variation)
                    audio_renders.append([idx, midi_note[0], midi_note[1], variation, audio])
                    m = np.abs(audio).max()
                    if m > max_amplitude:
                        max_amplitude = m
            # check for amplitude/normalization issues
            if max_amplitude > 0.99:
                warn_message = "Surge Patch idx={:04d} ({}): max audio amplitude = {} (all renders will be normalized)"\
                               .format(idx, self._synth.get_patch_info(idx), max_amplitude)
                # print(warn_message)
                with open(self._normalized_audio_file_path, 'a') as f:
                    f.write(warn_message + "\n")
                for i in range(len(audio_renders)):
                    audio_renders[i][4] = 0.99 * audio_renders[i][4] / max_amplitude
            # Write files
            for render in audio_renders:
                idx, pitch, vel, var, audio = tuple(render)
                soundfile.write(self._get_wav_file_path(self._synth.get_UID_from_index(idx), pitch, vel, var),
                                data=audio, samplerate=Fs)

