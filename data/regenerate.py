"""
Contains methods for (re)generating datasets of synthesizer sounds.

The current configuration from config.py will be used
"""

import sys
import pathlib
import importlib

from data.surgedataset import SurgeDataset



def gen_surge_dataset(regenerate_wav, regenerate_spectrograms_stats):
    """
    Approx audio rendering time: 1.5 minute for 2300 patches, 6 notes/patch (48-core CPU)
    Approx spectrograms computation time: TODO

    :param regenerate_wav:
    :param regenerate_spectrograms_stats:  FIXME gen spectrograms also
    :return:
    """
    importlib.reload(sys)
    sys.path.append(pathlib.Path(__file__).parent.parent)
    import config  # Dirty path trick to import config.py from project root dir
    importlib.reload(config)

    # WARNING: when computing stats, please make sure that *all* midi notes are available
    regenerate_spectrograms_stats = False  # TODO report approx time
    if regenerate_spectrograms_stats:
        assert len(config.model.midi_notes) > 1  # all MIDI notes (6?) must be used to compute stats

    # No label restriction, no normalization, etc...
    surge_dataset = SurgeDataset(note_duration=config.model.note_duration,
                                 midi_notes=config.model.midi_notes,
                                 multichannel_stacked_spectrograms=config.model.stack_spectrograms,
                                 n_fft=config.model.stft_args[0], fft_hop=config.model.stft_args[1],
                                 n_mel_bins=config.model.mel_bins, Fs=config.model.sampling_rate,
                                 spectrogram_normalization=None,  # No normalization: we want to compute stats
                                 spectrogram_min_dB=config.model.spectrogram_min_dB)
    # check_constrains_consistency=False)  # TODO implement constraints consistency checks
    if not regenerate_wav and not regenerate_spectrograms_stats:
        print(surge_dataset)  # All files must be pre-rendered before printing
        for i in range(100):
            test = surge_dataset[i]  # try get an item - for debug purposes

    if regenerate_wav:
        # WRITE ALL WAV FILES (approx. ??? Go)
        surge_dataset.generate_wav_files()
    if regenerate_spectrograms_stats:
        # whole-dataset stats (for proper normalization)
        surge_dataset.compute_and_store_spectrograms_stats()



if __name__ == "__main__":
    gen_surge_dataset(regenerate_wav=True, regenerate_spectrograms_stats=False)


