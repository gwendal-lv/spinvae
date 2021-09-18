"""
This file allows to render audio file and use the generated Surge dataset.
Data is formatted such that it is compatible with other datasets (Dexed synth, NSynth audio, ...).
However, this dataset does not provide Surge synthesis parameters.
"""

import torchaudio

from data import abstractbasedataset
from synth import surge


class SurgeDataset(abstractbasedataset.AudioDataset):
    def __init__(self, note_duration, n_fft, fft_hop, Fs,
                 midi_notes=((60, 100),), multichannel_stacked_spectrograms=False,
                 n_mel_bins=-1, mel_fmin=30.0, mel_fmax=11e3,  # FIXME 8kHz mel max
                 normalize_audio=False, spectrogram_min_dB=-120.0, spectrogram_normalization='min_max',
                 fx_bypass_level=surge.FxBypassLevel.ALL):
        """
        Class for rendering an audio dataset for the Surge synth. Can be used by a PyTorch DataLoader.

        Please refer to abstractbasedataset.AudioDataset for documentation about constructor arguments.

        :param fx_bypass_level: Describes how much Surge FX should be bypassed.

        """
        super().__init__(note_duration, n_fft, fft_hop, Fs, midi_notes, multichannel_stacked_spectrograms, n_mel_bins,
                         mel_fmin, mel_fmax, normalize_audio, spectrogram_min_dB, spectrogram_normalization)
        #self._synth = surge.Surge

    @property
    def synth_name(self):
        return "Surge"

    @property
    def total_nb_presets(self):
        pass  # FIXME

    def get_wav_file(self, preset_UID, midi_note, midi_velocity):
        pass  # FIXME


if __name__ == "__main__":
    print("coucou")

