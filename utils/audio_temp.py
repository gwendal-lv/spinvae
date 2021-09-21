"""
Additional classes and functions for audio.py
(e.g. some must be out of audio.py to allow multiprocessing/pickling)
"""
import os
import multiprocessing
from datetime import datetime

import numpy as np

import librosa


def dataset_samples_rms(dataset,  # no typing hint because of cross-referencing issue
                        num_workers=os.cpu_count(), print_analysis_duration=True):
    """ Computes a list of RMS frames for each audio file available from the given dataset.
    Also returns outliers (each outlier as a (UID, pitch, vel, var) tuple) if min/max values are given, else None.

    :returns: (rms_frames_list, outliers_list) """
    if print_analysis_duration:
        print("Starting dataset RMS computation...")
    t_start = datetime.now()
    audio_rms_frames, outliers = dataset_samples_rms_multiproc(dataset,
                                                                  dataset.valid_preset_UIDs,
                                                                  num_workers)
    delta_t = (datetime.now() - t_start).total_seconds()
    if print_analysis_duration:
        print("Dataset RMS computation finished. {:.1f} min total ({:.1f} ms / wav, {} audio files)"
              .format(delta_t / 60.0, 1000.0 * delta_t / dataset.nb_valid_audio_files, dataset.nb_valid_audio_files))
    return audio_rms_frames, outliers


def _dataset_samples_rms(worker_args):
    dataset, preset_UIDs = worker_args
    """ Auxiliary function for dataset_samples_rms: computes a single batch (multiproc or not). """
    audio_rms_frames = list()  # all RMS frames (1 array / audio file)
    for preset_UID in preset_UIDs:
        for midi_note in [(60, 85)]:  # FIXME
            midi_pitch, midi_vel = midi_note
            for variation in range(18):  # FIXME
                audio, Fs = dataset.get_wav_file(preset_UID, midi_pitch, midi_vel, variation=variation)
                audio_rms_frames.append(librosa.feature.rms(audio))
                # TODO check min/max
    return audio_rms_frames, None


def dataset_samples_rms_multiproc(dataset, preset_UIDs, num_workers):
    if num_workers < 1:
        num_workers = 1
    if num_workers == 1:
        audio_rms_frames, outliers = _dataset_samples_rms(dataset, preset_UIDs)
    else:
        split_preset_UIDs = np.array_split(preset_UIDs, num_workers)
        workers_args = [(dataset, UIDs) for UIDs in split_preset_UIDs]
        with multiprocessing.Pool(num_workers) as p:  # automatically closes and joins all workers
            results = p.map(_dataset_samples_rms, workers_args)
    return None, None  # FIXME
