"""
Contains methods for (re)generating datasets of synthesizer sounds.

The current configuration from config.py will be used (automatically imported from inside the functions).
"""

import sys
import pathlib
import importlib
import warnings
from datetime import datetime

from data import dataset
from data.abstractbasedataset import AudioDataset



def gen_surge_dataset(regenerate_wav: bool, regenerate_spectrograms: bool):
    """
    Approx audio rendering time:
        35 minutes (7ms/patch) for 2300 patches, 6 notes and 18 variations / patch (48-core CPU),
        Total 30 Go
    Approx spectrograms computation time:
        Compute and store:    Mel: 17min (4.1ms / spectrogram)   ;     STFT only: 5.5min (1.4ms/spec)
        Normalize and store: 1.6ms / spectrogram
        Total 60 Go (nfft 1024 hop 0256 mels 0257 : spectrograms are twice the size of wav files)

    If both params are set to False, the entire dataset will be read on 1 CPU (testing procedure)
    """
    importlib.reload(sys)
    sys.path.append(pathlib.Path(__file__).parent.parent)
    import config  # Dirty path trick to import config.py from project root dir
    importlib.reload(config)

    # WARNING: when computing stats, please make sure that *all* midi notes are available
    if regenerate_spectrograms:
        if len(config.model.midi_notes) <= 1:
            raise AssertionError("All MIDI notes (6?) must be used to compute spectrograms and stats")

    # No label restriction, etc...
    surge_dataset = dataset.SurgeDataset(** dataset.model_config_to_dataset_kwargs(config.model),
                                         data_augmentation=True,
                                         check_consistency=(not regenerate_wav) and (not regenerate_spectrograms))
    _gen_dataset(surge_dataset, regenerate_wav, regenerate_spectrograms)



def gen_nsynth_dataset(regenerate_json: bool, regenerate_spectrograms: bool):
    """
    Approx downloaded audio size: 30 Go?
        --> 39 Go with re-sorted JSON files and added symlinks (< 30s to compute and write all of them)

    Approx spectrograms computation time:
        Compute and store:    Mel: ????min (????ms / spectrogram)   ;     STFT only: ????min (????ms/spec)
        Normalize and store: ????ms / spectrogram
        Total ????????? Go (nfft 1024 hop 0256 mels 0257 : spectrograms are twice the size of wav files)

    If both params are set to False, the entire dataset will be read on 1 CPU (testing procedure)
    """
    importlib.reload(sys)
    sys.path.append(pathlib.Path(__file__).parent.parent)
    import config  # Dirty path trick to import config.py from project root dir
    importlib.reload(config)

    # WARNING: when computing stats, please make sure that *all* midi notes are available
    if regenerate_spectrograms:
        if len(config.model.midi_notes) <= 1:
            raise AssertionError("All MIDI notes (6?) must be used to compute spectrograms and stats")

    # No label restriction, etc...
    nsynth_dataset = dataset.NsynthDataset(** dataset.model_config_to_dataset_kwargs(config.model),
                                           data_augmentation=True,
                                           dataset_type='full',
                                           exclude_instruments_with_missing_notes=True,
                                           exclude_sonic_qualities=['reverb'],
                                           force_include_all_acoustic=True
                                           )
    print(nsynth_dataset)
    if regenerate_json:
        nsynth_dataset.regenerate_json_and_symlinks()
    #_gen_dataset(nsynth_dataset, False, regenerate_spectrograms)



def _gen_dataset(_dataset: AudioDataset, regenerate_wav: bool, regenerate_spectrograms: bool):
    if regenerate_wav:  # WRITE ALL WAV FILES
        _dataset.generate_wav_files()
    if regenerate_spectrograms:  # whole-dataset spectrograms and stats (for proper normalization)
        _dataset.compute_and_store_spectrograms_and_stats()

    if not regenerate_wav and not regenerate_spectrograms:  # Test : read the entire dataset
        print(_dataset)  # All files must be pre-rendered before printing
        t_start = datetime.now()
        for i in range(len(_dataset)):
            _ = _dataset[i]  # try get an item - for debug purposes
        delta_t = (datetime.now() - t_start).total_seconds()
        print("{} __getitem__ calls: {:.1f}s total, {:.1f}ms/call"
              .format(len(_dataset), delta_t, 1000.0 * delta_t / len(_dataset)))



if __name__ == "__main__":

    # gen_surge_dataset(regenerate_wav=False, regenerate_spectrograms=False)
    gen_nsynth_dataset(regenerate_json=False, regenerate_spectrograms=False)

