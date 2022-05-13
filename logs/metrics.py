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



class VectorMetric:
    """ Can be used to accumulate vector values during evaluation or training.
        dataset_len should be provided to improve performance (reduce memory allocations) """
    def __init__(self, dataset_len=-1):
        self.dataset_len = dataset_len
        if self.dataset_len < 0:
            raise NotImplementedError("[VectorMetric] Warning: not initialized with dataset_len "
                                      "- no memory pre-allocation (slow and currently not implemented)")
        self.on_new_epoch()

    def on_new_epoch(self):
        self._data = np.empty((0, 0))
        self.next_dataset_idx = 0

    def append(self, minibatch_data: torch.Tensor):
        if len(minibatch_data.shape) != 2:
            raise AssertionError("This class handles 2D data only (mini-batches of 1D vectors")
        if self.next_dataset_idx == 0:  # Memory allocation when the very first element is appended
            self._data = np.zeros((self.dataset_len, minibatch_data.shape[1]))
        batch_len = minibatch_data.shape[0]
        self._data[self.next_dataset_idx:(self.next_dataset_idx+batch_len), :] = \
            minibatch_data.detach().clone().cpu().numpy()
        self.next_dataset_idx += batch_len

    def get(self):
        if self.dataset_len > 0 and self.next_dataset_idx != self.dataset_len:
            raise AssertionError("Too few items were appended before calling this get() method")
        return self._data



class LatentMetric:
    """ Can be used to accumulate latent values during evaluation or training.
    dim_z and dataset_len should be provided to improve performance (reduce memory allocations) """
    def __init__(self, dim_z, dataset_len, dim_label=-1):
        self.dim_z = dim_z
        self.dataset_len = dataset_len
        self.valid_keys = ['mu', 'sigma', 'z0', 'zK']
        self.dim_label = dim_label
        if self.dim_label > 0:
            self.valid_keys.append('label')
        self.on_new_epoch()

    def on_new_epoch(self):
        self.next_dataset_index = 0  # Current row index to append data
        self._z, self._z_buf = dict(), dict()
        for k in self.valid_keys:
            if k != 'label':
                self._z_buf[k] = np.zeros((self.dataset_len, self.dim_z))
                self._z[k] = np.zeros((self.dataset_len, self.dim_z))
            else:  # 'label'
                self._z_buf[k] = np.zeros((self.dataset_len, self.dim_label), dtype=np.uint8)
                self._z[k] = np.zeros((self.dataset_len, self.dim_label), dtype=np.uint8)
        # Correlation matrices will be computed on-demand - reinit here to empty arrays
        self._spearman_corr_matrix = {'z0': np.zeros(0), 'zK': np.zeros(0)}
        self._spearman_corr_matrix_zerodiag = {'z0': np.zeros(0), 'zK': np.zeros(0)}
        self._avg_abs_corr_spearman_zerodiag = {'z0': -1.0, 'zK': -1.0}

    def append(self, z_mu_logvar, z0_samples, zK_samples=None, labels=None):
        """ Internally duplicates the latent values (and labels) of a minibatch """
        batch_len = z0_samples.shape[0]
        if zK_samples is None:
            zK_samples = z0_samples
        # Use pre-allocated storage matrices
        storage_rows = range(self.next_dataset_index, self.next_dataset_index+batch_len)
        self._z['mu'][storage_rows, :] = z_mu_logvar[:, 0, :].clone().detach().cpu().numpy()
        self._z['sigma'][storage_rows, :] = np.exp(z_mu_logvar[:, 1, :].clone().detach().cpu().numpy() * 0.5)
        self._z['z0'][storage_rows, :] = z0_samples.clone().detach().cpu().numpy()
        self._z['zK'][storage_rows, :] = zK_samples.clone().detach().cpu().numpy()
        if 'label' in self.valid_keys:
            if labels is None:
                raise ValueError("labels argument must be provided (dim_label was provided to class ctor)")
            else:
                self._z_buf['label'][storage_rows, :] = labels.clone().detach().cpu().numpy()
        else:
            if labels is not None:
                raise ValueError("labels argument cannot be provided because dim_label was not provided to class ctor")
        self.next_dataset_index += batch_len

    def get_z(self, z_type):
        """ Returns the requested latent values.

        :param z_type: 'mu', 'sigma', 'z0', 'zK' or 'latent'
        """
        if self.next_dataset_index != self.dataset_len:
            raise AssertionError("Internal storage matrices are not entirely set: {} items expected, {} items appended"
                                 .format(self.dataset_len, self.next_dataset_index))
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
        if not (z_type == 'z0' or z_type == 'zK'):
            raise AssertionError("Cannot compute correlation for latent data '{}'".format(z_type))
        self._spearman_corr_matrix[z_type], _ = scipy.stats.spearmanr(self.get_z(z_type))  # We don't use p-values
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

