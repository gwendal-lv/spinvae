"""
Contains methods for (re)generating datasets of synthesizer sounds.

The current configuration from config.py will be used (automatically imported from inside the functions).
"""

import sys
import pathlib
import importlib
import warnings
from datetime import datetime

import config

from data import dataset
from data.dataset import SurgeDataset, NsynthDataset, DexedDataset
from data.abstractbasedataset import AudioDataset

import synth.surge  # To re-generate the list of patches (included in the synth itself)

import utils.label



def gen_dexed_dataset(regen_wav: bool, regen_spectrograms: bool, regen_learnable_presets: bool, regen_labels: bool):
    """
    Approx audio rendering time:
        10.8 minutes (3.6ms/file) for 30293 patches, 6 notes and 1 variations / patch (48-core CPU),
        Total 44 GB
        --> w/ data augmentation (4 presets variations): 44 min (3.6ms/file), 727k files, 175 GB

    Approx spectrograms computation time: (2 variations)
        Compute and store:    Mel: 26.6 min (4.4 ms / spectrogram)   ;     STFT only: 8.3 min (1.4 ms/spec)
        Normalize and store: 1.7 ms / spectrogram
        Total / spectrogram config: 90 GB

    If both params are set to False, the entire dataset will be read on 1 CPU (testing procedure)
        (4.4 ms / __getitem__ call (6 notes / item ; 1 CPU) - live learnable representations computation)
        3.1 ms / __getitem__ call (6 notes / item ; 1 CPU) <- pre-computed learnable representations
        (1.0ms without preset learnable representations calculations <- )

    Learnable preset regeneration:
        1.3 min, 30293 presets with 1x data augmentation (presets variations), 2.7 ms / file  (on 1 CPU)
    """
    model_config, train_config = config.ModelConfig(), config.TrainConfig()
    config.update_dynamic_config_params(model_config, train_config)

    operators = model_config.dataset_synth_args[1]
    # continuous_params_max_resolution = 100

    # No label restriction, no normalization, etc...
    dexed_dataset = DexedDataset(
        ** dataset.model_config_to_dataset_kwargs(model_config),
        algos=None,  # allow all algorithms
        operators=operators,  # Operators limitation (config.py, or chosen above)
        restrict_to_labels=None,
        check_constrains_consistency=(not regen_wav) and (not regen_spectrograms)
    )
    if regen_learnable_presets:
        dexed_dataset.compute_and_store_learnable_presets()
    _gen_dataset(dexed_dataset, regen_wav, regen_spectrograms)
    if regen_labels:  # Instruments labels only
        labeler = utils.label.NameBasedLabeler(dexed_dataset)
        labeler.extract_labels(verbose=True)
        print(labeler)
        dexed_dataset.save_labels(labeler.instrument_labels, labeler.labels_per_UID)



def gen_surge_dataset(regen_patches_list: bool, regen_wav: bool, regen_spectrograms: bool, regen_labels: bool):
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
    # FIXME REFACTOR
    importlib.reload(sys)
    sys.path.append(pathlib.Path(__file__).parent.parent)
    import config  # Dirty path trick to import config.py from project root dir
    importlib.reload(config)

    if regen_patches_list:
        synth.surge.Surge.update_patches_list()

    # No label restriction, etc... FIXME also regenerate JSON patches list
    surge_dataset = SurgeDataset(** dataset.model_config_to_dataset_kwargs(config.model),
                                 data_augmentation=True,
                                 check_consistency=(not regen_wav) and (not regen_spectrograms))
    if regen_labels:  # Instruments labels only
        labeler = utils.label.SurgeReLabeler(surge_dataset)
        labeler.extract_labels(verbose=True)
        print(labeler)
        surge_dataset.save_labels(labeler.instrument_labels, labeler.labels_per_UID)

    _gen_dataset(surge_dataset, regen_wav, regen_spectrograms)


def gen_nsynth_dataset(regen_json: bool, regen_spectrograms: bool, regen_labels: bool):
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
    # FIXME REFACTOR
    importlib.reload(sys)
    sys.path.append(pathlib.Path(__file__).parent.parent)
    import config  # Dirty path trick to import config.py from project root dir
    importlib.reload(config)

    # No label restriction, etc... FIXME also regenerate labels
    nsynth_dataset = NsynthDataset(** dataset.model_config_to_dataset_kwargs(config.model),
                                   data_augmentation=True,
                                   dataset_type='full',
                                   exclude_instruments_with_missing_notes=True,
                                   exclude_sonic_qualities=None,#['reverb'],
                                   force_include_all_acoustic=True,
                                   required_midi_notes=config.model.required_dataset_midi_notes
                                   )
    if regen_json:
        nsynth_dataset.regenerate_json_and_symlinks()
    _gen_dataset(nsynth_dataset, False, regen_spectrograms)
    if regen_labels:  # Instruments labels only
        labeler = utils.label.NSynthReLabeler(nsynth_dataset)
        labeler.extract_labels(verbose=True)
        print(labeler)
        nsynth_dataset.save_labels(labeler.instrument_labels, labeler.labels_per_UID)


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

    gen_dexed_dataset(regen_wav=False, regen_spectrograms=False, regen_learnable_presets=False, regen_labels=False)
    #gen_surge_dataset(regen_patches_list=True, regen_wav=False, regen_spectrograms=False, regen_labels=True)
    #gen_nsynth_dataset(regen_json=False, regen_spectrograms=False, regen_labels=True)

