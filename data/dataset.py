"""
Datasets of synth sounds. PyTorch-compatible, with a lot of added method and properties for synthesizer
parameters learning.
Concrete preset Datasets are available from this module but are implemented in their own files.
"""


from . import dexeddataset
from . import surgedataset
from . import nsynthdataset

# ====================== Concrete dataset classes ======================
DexedDataset = dexeddataset.DexedDataset
SurgeDataset = surgedataset.SurgeDataset
NsynthDataset = nsynthdataset.NsynthDataset
# ======================================================================



def model_config_to_dataset_kwargs(model_config):
    """ Creates a dict that can be unpacked to pass to an AudioDataset class constructor.

    :param model_config: should be the config.model attribute from config.py. """
    return {'note_duration': model_config.note_duration, 'n_fft': model_config.stft_args[0],
            'fft_hop': model_config.stft_args[1], 'Fs': model_config.sampling_rate,
            'midi_notes': model_config.midi_notes, 'multichannel_stacked_spectrograms': model_config.stack_spectrograms,
            'n_mel_bins': model_config.mel_bins,
            'mel_fmin': model_config.mel_f_limits[0], 'mel_fmax': model_config.mel_f_limits[1],
            'spectrogram_min_dB': model_config.spectrogram_min_dB,
            'data_storage_root_path': model_config.data_root_path
            }
