"""
This file allows to render audio file and use the generated Surge dataset.
Data is formatted such that it is compatible with other datasets (Dexed synth, NSynth audio, ...).
However, this dataset does not provide Surge synthesis parameters.
"""
import multiprocessing
import pathlib
import json
import copy
from datetime import datetime
import os

import numpy as np
import torchaudio
import soundfile

from data import abstractbasedataset
from synth import surge


class SurgeDataset(abstractbasedataset.AudioDataset):
    def __init__(self, note_duration, n_fft, fft_hop, Fs,
                 midi_notes=((60, 100),), multichannel_stacked_spectrograms=False,
                 n_mel_bins=-1, mel_fmin=0, mel_fmax=8000,
                 normalize_audio=False, spectrogram_min_dB=-120.0, spectrogram_normalization='min_max',
                 fx_bypass_level=surge.FxBypassLevel.ALL,
                 data_storage_path="/media/gwendal/Data/Datasets/Surge"):
        """
        Class for rendering an audio dataset for the Surge synth. Can be used by a PyTorch DataLoader.

        Please refer to abstractbasedataset.AudioDataset for documentation about constructor arguments.

        :param fx_bypass_level: Describes how much Surge FX should be bypassed.
        :param data_storage_path: The absolute folder to store the generated datasets.
        """
        super().__init__(note_duration, n_fft, fft_hop, Fs, midi_notes, multichannel_stacked_spectrograms, n_mel_bins,
                         mel_fmin, mel_fmax, normalize_audio, spectrogram_min_dB, spectrogram_normalization)
        self._synth = surge.Surge(reduced_Fs=Fs, midi_note_duration_s=note_duration[0],
                                  render_duration_s=note_duration[0]+note_duration[1],
                                  fx_bypass_level=fx_bypass_level)
        self.data_storage_path = pathlib.Path(data_storage_path)
        # All available presets are considered valid
        self.valid_preset_UIDs = [self._synth.get_patch_info(idx)['UID'] for idx in range(self._synth.n_patches)]

    @property
    def synth_name(self):
        return "Surge"

    @property
    def total_nb_presets(self):
        return self._synth.n_patches

    def get_wav_file(self, preset_UID, midi_note, midi_velocity):
        pass  # FIXME

    def _get_patches_info_path(self):
        return self.data_storage_path.joinpath("dataset_info.json")

    def _get_audio_file_path(self, patch_UID, midi_pitch, midi_vel):  # TODO data augmentation args (note start/length)
        file_name = "{:04d}_pitch{:03d}vel{:03d}.wav".format(patch_UID, midi_pitch, midi_vel)
        return self.data_storage_path.joinpath("Audio").joinpath(file_name)

    def generate_wav_files(self):
        t_start = datetime.now()
        self.data_storage_path.joinpath("Audio").mkdir(parents=False, exist_ok=True)
        # 1) retrieve data for the .json file and for audio rendering
        dataset_info = {'surge': self._synth.dict_description,
                        'patches_count': self.valid_presets_count,
                        'patches': list()}
        valid_preset_indexes = list()
        for idx in range(self.total_nb_presets):
            p_info = self._synth.get_patch_info(idx)
            if p_info['UID'] in self.valid_preset_UIDs:  # We generate valid UIDs only
                p_info['midi_notes'] = self.midi_notes
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
        num_wav_written = len(self.valid_preset_UIDs) * len(self.midi_notes)
        print("Finished writing {} .wav files ({:.1f}s total, {:.1f}ms/file using {} CPUs)"
              .format(num_wav_written, delta_t, 1000.0*delta_t/num_wav_written, num_workers))

    def _generate_wav_files_batch(self, preset_indexes):
        for idx in preset_indexes:
            for midi_note in self.midi_notes:
                audio, Fs = self._synth.render_note(idx, midi_note[0], midi_note[1])
                soundfile.write(self._get_audio_file_path(self._synth.get_UID_from_index(idx),
                                                          midi_note[0], midi_note[1]),
                                data=audio, samplerate=Fs)
                # TODO check for amplitude/normalization issues.... pre-render all notes before normalizing ?



if __name__ == "__main__":
    pass


