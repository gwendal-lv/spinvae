"""
Easy-to-use metrics classes
"""

from collections import deque
import copy

import numpy as np
import scipy.stats

import torch


class BufferedMetric:
    """ Can store a limited number of metric values in order to get a smoothed estimate of the metric. """
    def __init__(self, buffer_size=10):
        self.buffer_size = buffer_size
        self.buffer = deque()

    def append(self, value):
        if isinstance(value, torch.Tensor):
            self.buffer.append(value.item())
        else:
            self.buffer.append(value)
        if len(self.buffer) > self.buffer_size:
            self.buffer.popleft()

    @property
    def mean(self):
        if len(self.buffer) == 0:
            raise ValueError()
        return np.asarray(self.buffer).mean()


class SimpleMetric:
    """ A very simple class for storing a metric, which provides EpochMetric-compatible methods """
    def __init__(self, value=0.0):
        if isinstance(value, torch.Tensor):
            self._value = value.item()
        else:
            self._value = value

    def on_new_epoch(self):
        return None

    def set(self, value):
        self._value = value

    def get(self):
        return self._value

    @property
    def value(self):
        return self.get()


class EpochMetric:
    """ Can store mini-batch metric values in order to compute an epoch-averaged metric. """
    def __init__(self, normalized_losses=True):
        """
        :param normalized_losses: If False, the mini-batch size must be given when data is appended
        """
        # :param epoch_end_metric: If given, this class will append end-of-epoch values to this BufferedMetric instance
        self.normalized_losses = normalized_losses
        self.buffer = list()

    def on_new_epoch(self):
        self.buffer = list()

    def append(self, value, minibatch_size=-1):
        if minibatch_size <= 0:
            assert self.normalized_losses is True
        if isinstance(value, torch.Tensor):
            self.buffer.append(value.item())
        else:
            self.buffer.append(value)

    def get(self):
        """ Returns the mean of values stored since last call to on_new_epoch() """
        if len(self.buffer) == 0:
            raise ValueError()
        return np.asarray(self.buffer).mean()

    @property
    def value(self):
        return self.get()


class LatentMetric:
    """ Can be used to accumulate latent values during evaluation or training.
    dim_z and dataset_len should be provided to improve performance (reduce memory allocations) """
    def __init__(self, dim_z=-1, dataset_len=-1):
        self.dim_z = dim_z
        self.dataset_len = dataset_len
        if self.dim_z < 0 or self.dataset_len < 0:
            raise AssertionError("[LatentMetric] Warning: not initialized with dim_z and dataset_len"
                                 "- no memory pre-allocation (slow)")
        self.on_new_epoch()

    def on_new_epoch(self):
        self.next_dataset_index = 0  # Current row index to append data
        self.valid_keys = ['mu', 'sigma', 'z0', 'zK']
        self._z, self._z_buf = dict(), dict()
        for k in self.valid_keys:
            if self.dim_z < 0 or self.dataset_len < 0:
                # Dicts of empty numpy arrays, that will be built once on the first plot method call
                self._z_buf[k] = np.zeros(0)
                self._z[k] = np.zeros(0)
            else:
                self._z_buf[k] = np.zeros((self.dataset_len, self.dim_z))
                self._z[k] = np.zeros((self.dataset_len, self.dim_z))
        self._spearman_corr_matrix = {'z0': np.zeros(0), 'zK': np.zeros(0)}
        self._spearman_corr_matrix_zerodiag = {'z0': np.zeros(0), 'zK': np.zeros(0)}
        self._avg_abs_corr_spearman_zerodiag = {'z0': -1.0, 'zK': -1.0}

    def append(self, z_mu_logvar, z0_samples, zK_samples=None):
        """
        Internally duplicates the latent values of a minibatch
        """
        if zK_samples is None:
            zK_samples = z0_samples
        # Tensor must be cloned before detach!  TODO tester copie sauvage?
        self._z_buf['mu'] = z_mu_logvar[:, 0, :].clone().detach().cpu().numpy()
        self._z_buf['sigma'] = np.exp(z_mu_logvar[:, 1, :].clone().detach().cpu().numpy() * 0.5)
        self._z_buf['z0'] = z0_samples.clone().detach().cpu().numpy()
        self._z_buf['zK'] = zK_samples.clone().detach().cpu().numpy()
        batch_len = self._z_buf['mu'].shape[0]
        if self.dim_z < 0 or self.dataset_len < 0:  # Dynamic memory allocation - slow
            if self._z['mu'].shape[0] == 0:  # Init?
                for k in self._z:  # k is 'mu', 'sigma', ...
                    self._z[k] = self._z_buf[k]
            else:
                for k in self._z:
                    self._z[k] = np.vstack((self._z[k], self._z_buf[k]))
        else:  # Pre-allocated storage matrices
            for k in self._z:
                self._z[k][self.next_dataset_index:self.next_dataset_index+batch_len, :] = self._z_buf[k]
            self.next_dataset_index += batch_len

    def get_z(self, z_type):
        """ Returns the requested latent values.

        :param z_type: 'mu', 'sigma', 'z0' or 'zK'
        """
        return self._z[z_type]

    def get_avg_abs_spearman_corr_zerodiag(self, z_type):  # TODO corr for z0 or zK ?
        """
        Returns the main metric of this class:
         the average of the Spearman absolute correlation matrix (with zeros on its diagonal).
         This function triggers the only epoch matrix computation, and should be called only once
         at epoch end.

        :param z_type: 'z0' or 'zK'
        """
        if self._spearman_corr_matrix[z_type].shape[0] == 0:  # Compute the matrix (auto-reset on new epoch) only once
            self._compute_correlation(z_type)
        return self._avg_abs_corr_spearman_zerodiag[z_type]

    def _compute_correlation(self, z_type):
        """
        :param z_type: 'z0' or 'zK'
        """
        assert(z_type == 'z0' or z_type == 'zK')
        self._spearman_corr_matrix[z_type], _ = scipy.stats.spearmanr(self._z[z_type])  # We don't use p-values
        self._spearman_corr_matrix_zerodiag[z_type] = copy.deepcopy(self._spearman_corr_matrix[z_type])
        for i in range(self._spearman_corr_matrix_zerodiag[z_type].shape[0]):
            self._spearman_corr_matrix_zerodiag[z_type][i, i] = 0.0
        self._avg_abs_corr_spearman_zerodiag[z_type] = np.abs(self._spearman_corr_matrix_zerodiag[z_type]).mean()

    def get_spearman_corr(self, z_type):
        """
        :param z_type: 'z0' or 'zK'
        """
        if self._spearman_corr_matrix[z_type].shape[0] == 0:
            self._compute_correlation(z_type)
        return self._spearman_corr_matrix[z_type]

    def get_spearman_corr_zerodiag(self, z_type):
        """
        :param z_type: 'z0' or 'zK'
        """
        if self._spearman_corr_matrix[z_type].shape[0] == 0:
            self._compute_correlation(z_type)
        return self._spearman_corr_matrix_zerodiag[z_type]


class LatentCorrMetric:
    """ Kind of Wrapper around LatentMetric to compute a Spearman correlation scalar """
    def __init__(self, latent_metric: LatentMetric, z_type: str):
        """
        :param z_type: 'z0' or 'zK'
        """
        self.latent_metric = latent_metric
        assert (z_type == 'z0' or z_type == 'zK')
        self.z_type = z_type

    def on_new_epoch(self):
        pass  # nothing to do (this class relies on LatentMetric)

    def get(self):
        return self.latent_metric.get_avg_abs_spearman_corr_zerodiag(self.z_type)


class CorrelationMetric:  # TODO merge into latent metric?
    def __init__(self, dim, dataset_len):
        self.data = np.empty((dataset_len, dim))
        self.observations_count = 0  # number of non-empty rows

    def append_batch(self, batch):
        batch_np = batch.clone().detach().cpu().numpy()
        start_idx = self.observations_count
        end_idx = self.observations_count + batch_np.shape[0]
        self.data[start_idx:end_idx, :] = batch_np
        self.observations_count = end_idx

    def get_spearman_corr_and_p_values(self):
        """
        Returns a tuple with the spearman r corr matrix and the corresponding p-values
        (null hypothesis H0: "two sets of data are uncorrelated")
        """
        assert self.observations_count == self.data.shape[0]  # All dataset elements must have been appended
        return scipy.stats.spearmanr(self.data, axis=0)  # observations in rows, variables in cols

