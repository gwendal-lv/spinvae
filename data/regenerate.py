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
from data.dataset import SurgeDataset, NsynthDataset, DexedDataset
from data.abstractbasedataset import AudioDataset



def gen_dexed_dataset(regenerate_wav: bool, regenerate_spectrograms: bool):
    """
    Approx audio rendering time:
        10.8 minutes (3.6ms/file) for 30200 patches, 6 notes and 1 variations / patch (48-core CPU),
        Total 44 Go
    Approx spectrograms computation time: (2 variations)
        Compute and store:    Mel: ??? min (??? ms / spectrogram)   ;     STFT only: 8.3 min (1.4 ms/spec)
        Normalize and store: 1.7 ms / spectrogram
        Total / spectrogram config: 90 Go

    If both params are set to False, the entire dataset will be read on 1 CPU (testing procedure)
        4.4 ms / __getitem__ call (6 notes / item ; 1 CPU)
        (1.0ms without preset learnable representations calculations <- )
    """
    importlib.reload(sys)
    sys.path.append(pathlib.Path(__file__).parent.parent)
    import config  # Dirty path trick to import config.py from project root dir
    importlib.reload(config)

    #operators = config.model.dataset_synth_args[1]  # Custom operators limitation?
    operators = None

    # No label restriction, no normalization, etc...
    # But: OPERATORS LIMITATIONS and DEFAULT PARAM CONSTRAINTS (main params (filter, transpose,...) are constant)
    dexed_dataset = DexedDataset(** dataset.model_config_to_dataset_kwargs(config.model),
                                 algos=None,  # allow all algorithms
                                 operators=operators,  # Operators limitation (config.py, or chosen above)
                                 # Params learned as categorical: maybe comment
                                 vst_params_learned_as_categorical=config.model.synth_vst_params_learned_as_categorical,
                                 restrict_to_labels=None,
                                 check_constrains_consistency=False)
    #print(dexed_dataset.preset_indexes_helper)
    _gen_dataset(dexed_dataset, regenerate_wav, regenerate_spectrograms)


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
        1.4ms / __getitem__ call (6 notes / item ; 1 CPU)
    """
    importlib.reload(sys)
    sys.path.append(pathlib.Path(__file__).parent.parent)
    import config  # Dirty path trick to import config.py from project root dir
    importlib.reload(config)

    # No label restriction, etc...
    surge_dataset = SurgeDataset(** dataset.model_config_to_dataset_kwargs(config.model),
                                 data_augmentation=True,
                                 check_consistency=(not regenerate_wav) and (not regenerate_spectrograms))
    _gen_dataset(surge_dataset, regenerate_wav, regenerate_spectrograms)


def gen_nsynth_dataset(regenerate_json: bool, regenerate_spectrograms: bool):
    """
    Approx downloaded audio size: 30 GB?
        --> 39 GB with re-sorted JSON files and added symlinks (< 30s to compute and write all of them)

    Approx spectrograms computation time:
        Compute and store:    Mel: ????min (????ms / spectrogram)   ;     STFT only: ????min (????ms/spec)
        Normalize and store: ????ms / spectrogram
        Total 1.1 GB (nfft 1024 hop 0256 mels 0257 : spectrograms are twice the size of wav files)
            for each spectrogram configuration

    If both params are set to False, the entire dataset will be read on 1 CPU (testing procedure)
        1.0ms / __getitem__ call (6 notes / item ; 1 CPU)
    """
    importlib.reload(sys)
    sys.path.append(pathlib.Path(__file__).parent.parent)
    import config  # Dirty path trick to import config.py from project root dir
    importlib.reload(config)

    # No label restriction, etc...
    nsynth_dataset = NsynthDataset(** dataset.model_config_to_dataset_kwargs(config.model),
                                   data_augmentation=True,
                                   dataset_type='full',
                                   exclude_instruments_with_missing_notes=True,
                                   exclude_sonic_qualities=None,#['reverb'],
                                   force_include_all_acoustic=True
                                   )
    if regenerate_json:
        nsynth_dataset.regenerate_json_and_symlinks()
    _gen_dataset(nsynth_dataset, False, regenerate_spectrograms)


def _gen_dataset(_dataset: AudioDataset, regenerate_wav: bool, regenerate_spectrograms: bool):
    print(_dataset)
    # When computing stats, please make sure that *all* midi notes are available
    if regenerate_wav or regenerate_spectrograms:
        if len(_dataset.midi_notes) <= 1:
            raise AssertionError("All MIDI notes (6?) must be used to compute spectrograms and stats")
    # --------------- WRITE ALL WAV FILES ---------------
    if regenerate_wav:
        _dataset.generate_wav_files()
    # ----- whole-dataset spectrograms and stats (for proper normalization) -----
    if regenerate_spectrograms:
        _dataset.compute_and_store_spectrograms_and_stats()

    if not regenerate_wav and not regenerate_spectrograms:
        print("Test: reading the entire dataset once...")
        t_start = datetime.now()
        for i in range(len(_dataset)):
            _ = _dataset[i]  # try get an item - for debug purposes
        delta_t = (datetime.now() - t_start).total_seconds()
        print("{} __getitem__ calls: {:.1f}s total, {:.1f}ms/call"
              .format(len(_dataset), delta_t, 1000.0 * delta_t / len(_dataset)))



if __name__ == "__main__":

    gen_dexed_dataset(regenerate_wav=False, regenerate_spectrograms=False)
    #gen_surge_dataset(regenerate_wav=False, regenerate_spectrograms=False)
    #gen_nsynth_dataset(regenerate_json=False, regenerate_spectrograms=False)

