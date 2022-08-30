import json
import pathlib
import pickle
import shutil
from typing import Optional, Dict, Any

import numpy as np
import os
import soundfile
import torch
from matplotlib import pyplot as plt

import utils.figures
from utils import timbre_librosa


class InterpSequence:
    def __init__(self, parent_path: pathlib.Path, seq_index: int):
        """ Class for loading/storing an interpolation sequence (audio and/or spectrograms output).

        :param parent_path: Path to the folder where all sequences are to be stored (each in its own folder).
        :param seq_index: Index (or ID) of that particular sequence; its own folder will be named after this index.
        """
        self.parent_path = parent_path
        self.seq_index = seq_index
        self.name = ''

        self.u = np.asarray([], dtype=float)
        self.UID_start, self.UID_end = -1, -1
        self.audio = list()
        self.spectrograms = list()
        self.librosa_features: Optional[Dict[str, Any]] = None
        # Actual number of frames might be greater (odd) because of window centering and signal padding
        self.num_metric_frames = 8  # 500ms frames for 4.0s audio  FIXME use ctor arg

    @property
    def storage_path(self) -> pathlib.Path:
        return self.parent_path.joinpath('{:05d}'.format(self.seq_index))

    def process_and_save(self, _librosa_features=False):
        """ Computes interpolation metrics
            then saves all available data into a new directory created inside the parent dir. """
        if _librosa_features:
            self.librosa_features = timbre_librosa.compute_audio_features(
                self.audio, self.num_metric_frames)
        if os.path.exists(self.storage_path):
            shutil.rmtree(self.storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=False)
        with open(self.storage_path.joinpath('info.json'), 'w') as f:
            json.dump({'sequence_index': self.seq_index, 'sequence_name': self.name,
                       'num_steps': len(self.audio), 'start': int(self.UID_start), 'end': int(self.UID_end)}, f)
        for step_i, audio_Fs in enumerate(self.audio):
            soundfile.write(self.storage_path.joinpath('audio_step{:02d}.wav'.format(step_i)),
                            audio_Fs[0], audio_Fs[1], subtype='FLOAT')
        with open(self.storage_path.joinpath('spectrograms.pkl'), 'wb') as f:
            pickle.dump(self.spectrograms, f)
        if self.librosa_features is not None:  # Don't save if librosa features were not computed
            with open(self.storage_path.joinpath('librosa_interp_metrics.dict.pkl'), 'wb') as f:
                pickle.dump(self.librosa_features, f)
        fig, axes = utils.figures.plot_spectrograms_interp(
            self.u, torch.vstack([torch.unsqueeze(torch.unsqueeze(s, dim=0), dim=0) for s in self.spectrograms]),
            metrics=self.librosa_features, plot_delta_spectrograms=False,
            title=(self.name if len(self.name) > 0 else None)
        )
        fig.savefig(self.storage_path.joinpath("spectrograms_interp.pdf"))
        plt.close(fig)  # FIXME???
        fig.savefig(self.storage_path.joinpath("spectrograms_interp.png"))
        plt.close(fig)

    def load(self):
        """ Loads a sequence using previously rendered data. """
        with open(self.storage_path.joinpath('info.json'), 'r') as f:
            json_info = json.load(f)
            self.UID_start, self.UID_end, self.name, num_steps = \
                json_info['start'], json_info['end'], json_info['sequence_name'], json_info['num_steps']
        # FIXME don't always load these interp metrics
        with open(self.storage_path.joinpath('librosa_interp_metrics.dict.pkl'), 'rb') as f:
            self.librosa_features = pickle.load(f)
        with open(self.storage_path.joinpath('spectrograms.pkl'), 'rb') as f:
            self.spectrograms = pickle.load(f)
        self.audio = list()
        for step_i in range(num_steps):
            self.audio.append(soundfile.read(self.storage_path.joinpath('audio_step{:02d}.wav'.format(step_i))))

    # TODO render independent spectrograms to PNG (for future website)


class LatentInterpSequence(InterpSequence):
    def __init__(self, parent_path: pathlib.Path, seq_index: int):
        super().__init__(parent_path=parent_path, seq_index=seq_index)
        self.z = torch.empty((0, ))
