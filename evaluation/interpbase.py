"""
Base class to compute metrics about an abstract interpolation method
"""
import os.path
import pathlib
import pickle
import shutil
from abc import ABC, abstractmethod
from typing import Optional, Tuple
from datetime import datetime

import numpy as np
import scipy.interpolate
import torch

import data
import evaluation.load


class InterpSequence:
    def __init__(self, parent_path: pathlib.Path, name: str):
        """ Class for loading/storing an interpolation sequence (audio and/or spectrograms output).

        :param parent_path: Path to the folder where all sequences are to be stored (each in its own folder).
        :param name: Name of that particular sequence; its own folder will start by its name.
        """
        self.parent_path = parent_path
        self.name = name

        self.u = np.asarray([], dtype=float)
        self.UID_start, self.UID_end = -1, -1
        self.audio = list()
        self.spectrograms = torch.empty((0, ))

    def storage_path(self):
        return self.parent_path.joinpath(self.name)


class LatentInterpSequence(InterpSequence):
    def __init__(self, parent_path: pathlib.Path, name: str):
        super().__init__(parent_path=parent_path, name=name)
        self.z = torch.empty((0, ))


class InterpBase(ABC):
    def __init__(self, num_steps=7, u_curve='linear', verbose=True):
        """

        :param num_steps:
        :param u_interp_kind: The type of curve used for the interpolation abscissa u.
        """
        self.num_steps = num_steps
        self.u_curve = u_curve
        self.verbose = verbose

    def get_u_interpolated(self):
        if self.u_curve == 'linear':
            return np.linspace(0.0, 1.0, self.num_steps, endpoint=True)
        else:
            raise NotImplementedError()

    @property
    @abstractmethod
    def storage_path(self) -> pathlib.Path:
        pass

    # TODO a method to compute metrics about a given interpolation sequence

    # TODO a method to store a given interpolation (results as well as audio files, spectrograms and figures)
    #     generate and save independent figures (easier to show in HTML tables // github pages)

    # TODO accept spectrograms and/or audio as input


class ModelBasedInterpolation(InterpBase):
    def __init__(self, model_loader: Optional[evaluation.load.ModelLoader] = None,
                 device='cpu', num_steps=7,
                 latent_interp_kind='linear', verbose=True):  # TODO which MIDI note to use as default? channel index and raw MIDI
        """
        A class for performing interpolations using a neural network model whose inputs are latent vectors.


        :param model_loader: If given, most of other arguments (related to the model and corresponding
         dataset) will be ignored.
        """
        super().__init__(num_steps=num_steps, verbose=verbose)
        self.latent_interp_kind = latent_interp_kind
        if model_loader is not None:
            self._model_loader = model_loader
            self.device = model_loader.device
            self.dataset = model_loader.dataset
            self.dataset_type = model_loader.dataset_type
            self.dataloader, self.dataloader_num_items = model_loader.dataloader, model_loader.dataloader_num_items
            self._storage_path = model_loader.path_to_model_dir
            # extended_ae_model and/or reg_model may be None
            self.extended_ae_model, self.ae_model, self.reg_model \
                = model_loader.extended_ae_model, model_loader.ae_model, model_loader.reg_model
        else:
            self.device = device
            self.dataset, self.dataset_type, self.dataloader, self.dataloader_num_items = None, None, None, None
            self._storage_path = None

    @property
    def storage_path(self) -> pathlib.Path:
        if self._storage_path is not None:
            return self._storage_path.joinpath('interp_{}'.format(self.dataset_type))
        else:
            raise AssertionError("self._storage_path was not set (this instance was not built using a ModelLoader)")

    def process_dataset(self):
        """ Performs an interpolation over the whole given dataset (usually validation or test), using pairs
        of items from the dataloader. Dataloader should be deterministic. Total number of interpolations computed:
        len(dataloder) // 2. """
        # First: create the dir to store data (erase any previously written eval files)
        if os.path.exists(self.storage_path):
            shutil.rmtree(self.storage_path)
        os.mkdir(self.storage_path)
        t_start = datetime.now()
        if self.verbose:
            print("[{}] Results will be stored in '{}'".format(type(self).__name__, self.storage_path))

        # TODO store all stats (each sequence is written to SSD before computing the next one)
        # encoded latent vectors (usually different from endpoints, if the corresponding preset is not 100% accurate)
        z_ae = list()
        # Interpolation endpoints
        z_endpoints = list()

        current_sequence_index = 0
        # Retrieve all latent vectors that will be used for interpolation
        # We assume that batch size is an even number...
        for batch_idx, sample in enumerate(self.dataloader):
            end_sequence_with_next_item = False  # If True, the next item is the 2nd (last) of an InterpSequence
            for i in range(sample[0].shape[0]):
                x_in, v_target, sample_info = sample[0], sample[1], sample[2]
                if not end_sequence_with_next_item:
                    end_sequence_with_next_item = True
                else:
                    # It's easier to compute interpolations one-by-one (all data might not fit into RAM)
                    seq = LatentInterpSequence(self.storage_path, '{:05d}'.format(current_sequence_index))
                    seq.UID_start, seq.UID_end = sample_info[i-1, 0], sample_info[i, 0]
                    z_start, z_start_first_guess, acc, L1_err \
                        = self.compute_latent_vector(x_in[i-1:i, :], sample_info[i-1:i, :], v_target[i-1:i, :])
                    if acc < 100.0 or L1_err > 0.0:
                        print(acc, L1_err)
                    z_end, z_end_first_guess, acc, L1_err \
                        = self.compute_latent_vector(x_in[i:i+1, :], sample_info[i:i+1, :], v_target[i:i+1, :])
                    if acc < 100.0 or L1_err > 0.0:
                        print(acc, L1_err)
                    z_endpoints.append(z_start), z_endpoints.append(z_end)
                    z_ae.append(z_start_first_guess), z_ae.append(z_end_first_guess)
                    u, z = self.interpolate_latent(z_start, z_end)

                    current_sequence_index += 1
                    end_sequence_with_next_item = False

            # FIXME TEMP
            if batch_idx >= 1:
                break

        all_z_ae = torch.vstack(z_ae).detach().clone().cpu().numpy()
        all_z_endpoints = torch.vstack(z_endpoints).detach().clone().cpu().numpy()
        with open(self.storage_path.joinpath('all_z_ae.np.pkl'), 'wb') as f:
            pickle.dump(all_z_ae, f)
        with open(self.storage_path.joinpath('all_z_endpoints.np.pkl'), 'wb') as f:
            pickle.dump(all_z_endpoints, f)
        if self.verbose:
            delta_t = (datetime.now() - t_start).total_seconds()
            print("[{}] Finished processing {} interpolations in {:.1f}min total ({:.1f}s / interpolation)"
                  .format(type(self).__name__, all_z_endpoints.shape[0] // 2, delta_t / 60.0,
                          delta_t / (all_z_endpoints.shape[0] // 2)))

    @abstractmethod
    def compute_latent_vector(self, x, sample_info, v_target):
        """ Computes the most appropriate latent vector (child class implements this method) """
        pass

    def interpolate_latent(self, z_start, z_end) -> Tuple[np.ndarray, torch.Tensor]:
        """ Returns a N x D tensor of interpolated latent vectors, where N is the number of interpolation steps (here:
        considered as a batch size) and D is the latent dimension. Each latent coordinate is interpolated independently.

        Non-differentiable: based on scipy.interpolate.interp1d.

        :param z_start: 1 x D tensor
        :param z_end: 1 x D tensor
        :returns: u, interpolated_z
        """
        # TODO try RBF interpolation instead of interp 1d ?
        z_cat = torch.cat([z_start, z_end], dim=0)
        interp_f = scipy.interpolate.interp1d(
            [0.0, 1.0], z_cat.clone().detach().cpu().numpy(), kind=self.latent_interp_kind, axis=0,
            bounds_error=True)  # extrapolation disabled, no fill_value
        u_interpolated = self.get_u_interpolated()
        return u_interpolated, torch.tensor(interp_f(u_interpolated), device=self.device, dtype=torch.float32)


    # TODO also implement the naïve interpolation
