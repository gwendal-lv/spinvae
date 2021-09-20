"""
Contains methods for (re)generating datasets of synthesizer sounds.

The current configuration from config.py will be used (automatically imported from inside the functions).
"""

import sys
import pathlib
import importlib
import warnings

from data.surgedataset import SurgeDataset



def gen_surge_dataset(regenerate_wav, regenerate_spectrograms):
    """
    Approx audio rendering time:
        35 minutes (7ms/patch) for 2300 patches, 6 notes and 18 variations / patch (48-core CPU),
        Total 30 Go
    Approx spectrograms computation time:
        Compute and store: 17min (4.1ms / spectrogram)
        Normalize and store: 1.6ms / spectrogram
        Total 60 Go (nfft 1024 hop 0256 mels 0257 : spectrograms are twice the size of wav files)

    :param regenerate_wav:
    :param regenerate_spectrograms:
    :return:
    """
    importlib.reload(sys)
    sys.path.append(pathlib.Path(__file__).parent.parent)
    import config  # Dirty path trick to import config.py from project root dir
    importlib.reload(config)

    # WARNING: when computing stats, please make sure that *all* midi notes are available
    if regenerate_spectrograms:
        if len(config.model.midi_notes) <= 1:
            raise AssertionError("All MIDI notes (6?) must be used to compute spectrograms and stats")

    # No label restriction, no normalization, etc...
    surge_dataset = SurgeDataset(note_duration=config.model.note_duration,
                                 midi_notes=config.model.midi_notes,
                                 multichannel_stacked_spectrograms=config.model.stack_spectrograms,
                                 n_fft=config.model.stft_args[0], fft_hop=config.model.stft_args[1],
                                 n_mel_bins=config.model.mel_bins, Fs=config.model.sampling_rate,
                                 spectrogram_min_dB=config.model.spectrogram_min_dB)
    # check_constrains_consistency=False)  # TODO implement constraints consistency checks
    if not regenerate_wav and not regenerate_spectrograms:
        print(surge_dataset)  # All files must be pre-rendered before printing
        for i in range(100):
            test = surge_dataset[i]  # try get an item - for debug purposes

    if regenerate_wav:
        # WRITE ALL WAV FILES (approx. ??? Go)
        surge_dataset.generate_wav_files()
    if regenerate_spectrograms:
        # whole-dataset spectrograms and stats (for proper normalization)
        surge_dataset.compute_and_store_spectrograms_and_stats()



if __name__ == "__main__":

    gen_surge_dataset(regenerate_wav=False, regenerate_spectrograms=True)


