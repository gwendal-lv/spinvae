"""
Utilities for plotting various figures (spectrograms, ...)
"""
import pickle
import warnings
from typing import Optional, Sequence, Iterable, List, Dict

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mpl_colors
import seaborn as sns
import pandas as pd
import librosa.display
import torch
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix

import logs.metrics

from data.abstractbasedataset import AudioDataset, PresetDataset
from data.preset import PresetIndexesHelper
from data.preset2d import Preset2d, Preset2dHelper

import utils.stat


# Display parameters for scatter/box/error plots
__param_width = 0.12
__x_tick_font_size = 8
__max_nb_spec_presets = 4  # Max number of different presets whose corresponding spectrograms are displayed


def plot_audio(audio: np.ndarray, dataset: Optional[AudioDataset] = None, preset_UID: Optional[int] = None,
               figsize=(8, 3),
               Fs: Optional[int] = None, title: Optional[str] = None,
               midi_note: Optional[tuple] = None, variation: Optional[int] = None):
    if dataset is not None:  # if dataset is available: some input args are forgotten
        Fs = dataset.Fs
        if preset_UID is not None:
            title = "{} (UID={})".format(dataset.get_name_from_preset_UID(preset_UID), preset_UID)
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    if Fs is not None:
        t = np.linspace(0, audio.shape[0] * 1.0 / Fs, audio.shape[0])
    else:
        t = np.arange(audio.shape[0])
    plt.plot(t, audio)
    if title is None:
        plt.title("Waveform")
    else:
        plt.title(title)
    plt.xlabel("Time (s)" if Fs is not None else 'Sample')
    plt.ylabel("Amplitude")
    if midi_note is not None:
        legend_str = "MIDI: {}".format(midi_note)
        if variation is not None:
            legend_str += "\nvariation {}".format(variation)
        plt.legend([legend_str])
    fig.tight_layout()
    return fig, ax


def plot_dataset_item(dataset: Optional[AudioDataset], item_index: int,
                      figsize=(5, 4), add_colorbar=True, midi_note_index=-1):
    """
    Plots the spectrogram of an item from the given dataset, and loads the corresponding audio.
    :returns: fig, axes, audio, Fs
    """
    item_UID = dataset.valid_preset_UIDs[item_index]
    name = dataset.get_name_from_preset_UID(item_UID, long_name=True)
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    if midi_note_index < 0:
        midi_pitch, midi_vel = dataset.default_midi_note
    else:
        midi_pitch, midi_vel = dataset.midi_notes[midi_note_index]
    spectrogram = torch.load(dataset.get_spec_file_path(item_UID, midi_pitch, midi_vel)).numpy()
    im = librosa.display.specshow(spectrogram, shading='flat', ax=ax, cmap='magma')
    if add_colorbar:
        clb = fig.colorbar(im, ax=ax, orientation='vertical')
    ax.set(title="[{}] '{}'".format(item_UID, name))
    audio, Fs = dataset.get_wav_file(item_UID, midi_pitch, midi_vel)
    return fig, ax, audio, Fs


def plot_train_spectrograms(x_in, x_out, uid, notes, dataset: AudioDataset,
                            model_config, train_config):
    """ Wrapper for plot_spectrograms, which is made to be easily used during training/validation. """
    if dataset.multichannel_stacked_spectrograms:  # Plot only 1 preset ID, multiple midi notes
        midi_notes = notes[0, :, :]
        presets_names = [dataset.get_name_from_preset_UID(uid[0].item())]  # we send a single name
        max_nb_specs = midi_notes.shape[0]
    else:  # Plot multiple preset IDs
        midi_notes = notes[:, 0, :]  # FIXME which midi note to plot (default first, lowest pitch)
        presets_names = list()
        for i in range(np.minimum(train_config.logged_samples_count, uid.shape[0])):
            presets_names.append(dataset.get_name_from_preset_UID(uid[i].item()))
        max_nb_specs = len(presets_names)
    return plot_spectrograms(x_in, x_out, presets_UIDs=uid,
                             midi_notes=midi_notes,
                             multichannel_spectrograms=dataset.multichannel_stacked_spectrograms,
                             plot_error=(x_out is not None), max_nb_specs=max_nb_specs,
                             add_colorbar=True,
                             presets_names=presets_names)


def plot_spectrograms(specs_GT, specs_recons=None,
                      presets_UIDs=None, midi_notes=None, multichannel_spectrograms=False,
                      plot_error=False, error_magnitude=1.0,
                      max_nb_specs: Optional[int] = None,
                      spec_ax_w=2.5, spec_ax_h=3.0,
                      add_colorbar=False,
                      presets_names: Optional[List[str]] = None):
    """
    Creates a figure and axes to plot some ground-truth spectrograms (1st row) and optional reconstructed
    spectrograms (2nd row).
    If MIDI notes are not given or spectrograms are 1-ch, spectrograms from different preset UIDs will be plotted.
    If MIDI notes are given, different spectrograms of a single preset UIDs will be plotted.

    :param specs_GT: Tensor: batch of ground-truth spectrograms; 1-channel or multi-channel (multiple MIDI notes)
    :param specs_recons: Tensor: batch of reconstructed spectrograms
    :param presets_UIDs: 1d-Tensor of preset UIDs corresponding to the spectrograms to be plotted
    :param midi_notes: Tuple of (midi_pitch, midi_velocity) tuples which correspond to the spectrograms to be plotted
    :param multichannel_spectrograms: If True, will plot the channels of batch index 0. If False, will plot
        channel 0 of multiple spectrograms. midi_notes and presets_UIDs arguments must be coherent with this.
    :param error_magnitude: Max error magnitude (used to set the error spectrogram colorbar limits to -mag, +mag)
    :param spec_ax_w: width (in figure units) of a single spectrogram

    :returns: fig, axes
    """
    max_nb_specs = __max_nb_spec_presets if max_nb_specs is None else max_nb_specs
    if multichannel_spectrograms:  # Plot several notes (different channels)
        nb_specs = np.minimum(max_nb_specs, len(midi_notes))
    else:  # Plot several batch items, channel 0 only
        nb_specs = np.minimum(max_nb_specs, specs_GT.size(0))
    if add_colorbar:
        spec_ax_w *= 1.3
    if specs_recons is None:
        assert plot_error is False  # Cannot plot error without a reconstruction to be compared
        fig, axes = plt.subplots(1, nb_specs, figsize=(nb_specs*spec_ax_w, spec_ax_h))
        axes = [axes]  # Unsqueeze
        nb_rows = 1
    else:
        nb_rows = 2 if not plot_error else 3
        fig, axes = plt.subplots(nb_rows, nb_specs, figsize=(nb_specs*spec_ax_w, spec_ax_h*nb_rows))
    if nb_specs == 1:
        axes = [axes]
    # Row-by-row plots
    for row in range(nb_rows):
        for j in range(nb_specs):
            batch_idx, ch = (0, j) if multichannel_spectrograms else (j, 0)
            if row == 0:
                spectrogram = specs_GT[batch_idx, ch, :, :].clone().detach().cpu().numpy()
            elif row == 1:
                spectrogram = specs_recons[batch_idx, ch, :, :].clone().detach().cpu().numpy()
            else:
                spectrogram = specs_recons[batch_idx, ch, :, :].clone().detach().cpu().numpy()\
                              - specs_GT[batch_idx, ch, :, :].clone().detach().cpu().numpy()
            if row == 0:  # Some titles on first row only
                title = ''
                if multichannel_spectrograms and midi_notes is not None:
                    title = "note ({},{})".format(midi_notes[j][0], midi_notes[j][1])
                elif not multichannel_spectrograms:
                    if presets_UIDs is not None:
                        title += "{}".format(presets_UIDs[j].item())
                    if presets_names is not None:
                        title += " '{}'".format(presets_names[j])
                axes[row][j].set(title=title)
            im = librosa.display.specshow(spectrogram, shading='flat', ax=axes[row][j],
                                          cmap=('magma' if row < 2 else 'bwr'),
                                          vmin=(-error_magnitude if row == 2 else None),
                                          vmax=(error_magnitude if row == 2 else None))
            if add_colorbar:
                clb = fig.colorbar(im, ax=axes[row][j], orientation='vertical')

    if multichannel_spectrograms:
        title = ''
        if presets_UIDs is not None:
            title += 'UID={}'.format(presets_UIDs[0].item())
        if presets_names is not None:
            title += " '{}'".format(presets_names[0])
        if len(title) > 0:
            fig.suptitle(title)
    elif not multichannel_spectrograms and midi_notes is not None:
        fig.suptitle("note ({},{})".format(midi_notes[0][0], midi_notes[0][1]))
    fig.tight_layout()
    return fig, axes


def remove_axes_spines_and_ticks(ax):
    for spine in ['top', 'bottom', 'right', 'left']:
        ax.spines[spine].set_visible(False)
    ax.get_xaxis().set_ticks([])
    ax.get_yaxis().set_ticks([])


def plot_spectrograms_interp(
        u: np.ndarray, spectrograms: torch.Tensor, plot_delta_spectrograms=True,
        z: Optional[torch.Tensor] = None, num_z_coords_to_show=40, metrics: Optional[Dict[str, np.ndarray]] = None,
        subplot_w_h=(1.5, 1.5), title: Optional[str] = None):
    """
    Plots a "batch" of interpolated spectrograms. The number of spectrograms must be the same as the number of
    interpolation abscissae u.

    :param metrics: TODO DOC
    """
    if u.shape[0] != spectrograms.shape[0] or (z is not None and u.shape[0] != z.shape[0]):  # TODO all tests
        raise AssertionError("All input arrays/tensor must have the same length along dimension 0.")
    if u.shape[0] < 2:
        raise ValueError("This function requires 2 spectrograms or more")
    if spectrograms.shape[1] > 1:
        raise ValueError("This function supports single-channel spectrograms only.")
    n_rows = 2 if plot_delta_spectrograms else 1
    n_rows += (1 if z is not None else 0)
    n_rows += (1 if metrics is not None else 0)
    fig, axes = plt.subplots(n_rows, u.shape[0], sharey='row',
                             figsize=(u.shape[0]*subplot_w_h[0], n_rows*subplot_w_h[1] + (0 if title is None else 0.3)))
    if n_rows == 1:  # Always convert to 2D axes (even if single-line plot)
        axes = np.expand_dims(axes, axis=0)
    # Data converted to numpy, compute deltas
    if spectrograms.shape[1] > 1:
        raise AssertionError("Input spectrograms must be single-channel.")
    specs_np = spectrograms[:, 0, :, :].clone().detach().cpu().numpy()
    if plot_delta_spectrograms:
        delta_specs_np = np.zeros_like(specs_np)
        for u_idx in range(1, u.shape[0]):  # First delta will remain zero
            delta_specs_np[u_idx, :, :] = (specs_np[u_idx, :, :] - specs_np[u_idx-1, :, :]) / (u[u_idx] - u[u_idx-1])
    z_np = z.clone().detach().cpu().numpy() if z is not None else None
    z_indices_range = None
    if z_np is not None:
        if 0 < num_z_coords_to_show < z_np.shape[1]:
            z_indices_range = np.arange(z_np.shape[1] - num_z_coords_to_show, z_np.shape[1])
            z_np = z_np[:, z_indices_range]
        else:
            z_indices_range = np.arange(z_np.shape[1])
    # Prepare merged axes to plot interpolation metrics
    #    (https://matplotlib.org/stable/gallery/subplots_axes_and_figures/gridspec_and_subplots.html)
    m_axes = None
    if metrics is not None:  # Always on the last row
        row = n_rows - 1
        gs = axes[row, 0].get_gridspec()
        for ax in axes[row, :]:
            ax.remove()
        m_axes = fig.add_subplot(gs[row, :])
    # Plots
    delta_specs_cbar_ax = None
    for u_idx in range(u.shape[0]):
        _u = u[u_idx]
        axes[0, u_idx].set_title(r'$u=$' + '{:.2f}'.format(_u))
        # Spectrograms
        im = librosa.display.specshow(specs_np[u_idx, :, :], shading='flat', ax=axes[0, u_idx], cmap='magma',
                                      vmin=specs_np.min(), vmax=specs_np.max())
        if u_idx > 0 and plot_delta_spectrograms:
            max_abs_error = max(np.abs(delta_specs_np.min()), np.abs(delta_specs_np.max()))
            im = librosa.display.specshow(delta_specs_np[u_idx, :, :], shading='flat', ax=axes[1, u_idx], cmap='bwr',
                                          vmin=-max_abs_error, vmax=max_abs_error)
            if u_idx == 1:
                delta_specs_cbar_ax = fig.add_axes(axes[1, 0].get_position())
                clb = fig.colorbar(im, cax=delta_specs_cbar_ax)
        # Latent vectors  TODO display min/max values?
        if z_np is not None:
            colors = [mpl_colors.hsv_to_rgb([((c * 63.2) % 100)/100.0, 1.0, 0.85]) for c in z_indices_range]
            axes[2, u_idx].scatter(z_indices_range, z_np[u_idx, :], s=1, c=colors)
            axes[2, u_idx].grid(axis='y')
    # Interpolation metrics (pre-computed, display each dict key as a line)
    #     Normalize each metric before display (we don't care about the scale)
    if metrics is not None:
        legend_keys = list()
        row = 0
        for k, v in metrics.items():
            if len(v.shape) == 1:  # 1D metrics only
                legend_keys.append(k)
                normalized_values = v - v.min()
                normalized_values = normalized_values / normalized_values.max()
                m_axes.plot(np.arange(v.shape[0]), normalized_values, linestyle=('-' if row < 10 else '--'))
                m_axes.set_xlim([-0.5, v.shape[0] - 0.5])
                row += 1
        m_axes.legend(legend_keys, prop={'size': 5})
        m_axes.get_yaxis().set_visible(False)
        lol = 0
    # Display: labels, ...
    axes[0, 0].set_ylabel(r'Spectrogram $\mathbf{s}$')
    if n_rows > 1:
        remove_axes_spines_and_ticks(axes[1, 0])
        axes[1, 0].set_ylabel(r'$\frac{\Delta \mathbf{s}}{\Delta u}$', fontsize=14.0).set_rotation(0.0)
    if z_np is not None:
        axes[2, 0].set_ylabel(r'Latent vector $\mathbf{z}$')
    if title is not None:
        fig.suptitle(title)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fig.tight_layout()  # 'UserWarning: This figure includes Axes that are not compatible with tight_layout, ...'
    # Re-set colorbar axes positions, after layout has been tightened
    if delta_specs_cbar_ax is not None:
        axes_to_reposition = [(delta_specs_cbar_ax, axes[1, 0])]  # TODO ajouter autres axes à repos (si non-None)
        for cb_ax, original_ax in axes_to_reposition:
            pos = original_ax.get_position().bounds  # (x0, y0, width, height)
            cb_ax.set_position([pos[0] + pos[2]*0.45, pos[1], pos[2]*0.1, pos[3]])
    return fig, axes


def plot_vector_distributions_stats(metrics: Dict[str, logs.metrics.VectorMetric], metric_name: str):
    """

    :param metrics:
    :param metric_name: The base name of the metric to be plotted. 'metric_name/Train' and 'metric_name/Valid'
        keys must be found in the dict.
    :return:
    """
    num_coord_to_plot = 50
    gen_plot_eq_num_coord = 5
    fig, axes = plt.subplots(2, 3, figsize=(1.1 * __param_width * (num_coord_to_plot + 2*gen_plot_eq_num_coord), 4),
                             gridspec_kw={'width_ratios': [gen_plot_eq_num_coord, gen_plot_eq_num_coord,
                                                           num_coord_to_plot - 2*gen_plot_eq_num_coord]})
    # TODO KDE / histplots ????? et séparer >0 et <0
    # Retrieve data (non-flattened), per dataset
    data = {'Train': metrics[metric_name + '/Train'].get(), 'Valid': metrics[metric_name + '/Valid'].get()}
    flierprops = dict(marker='.', markerfacecolor='k', markersize=0.5, markeredgecolor='none')
    for row, ds_type in enumerate(['Train', 'Valid']):
        # Plot flattened data
        axes[row, 0].boxplot(x=data[ds_type].flatten(), vert=True, sym='.k', flierprops=flierprops)
        axes[row, 0].set(ylabel=ds_type, title="All datapoints")
        # plot flattened data, without outliers
        axes[row, 1].hist(x=utils.stat.remove_outliers(data[ds_type].flatten()), orientation='horizontal', bins=25)
        axes[row, 1].set(title="(w/o outliers)")
        # plot the 50 first per-coordinate output distributions
        sns.violinplot(data=data[ds_type][:, 0:num_coord_to_plot], ax=axes[row, 2], inner='quartile',
                       linewidth=0.5)
        axes[row, 2].set(title="Per-coordinate distributions ({} to {} not represented)"
                               .format(num_coord_to_plot, data[ds_type].shape[1]-1))

    fig.tight_layout()
    _configure_params_plot_x_tick_labels(axes[0, 2])
    _configure_params_plot_x_tick_labels(axes[1, 2])
    return fig, axes


def plot_latent_distributions_stats(latent_metric: logs.metrics.LatentMetric, figsize=None, eps=1e-7,
                                    max_displayed_latent_coords=140):
    """ Uses boxplots to represent the distribution of the mu and/or sigma parameters of
    latent gaussian distributions. Also plots a general histogram of all samples.

    :param latent_metric:
    :param figsize:
    :param eps: Value added to bounds for numerical stability (when a latent value is constant).
    """
    metrics_names = ['mu', 'sigma', 'zK']
    data_limited = [dict(), dict()]  # We display the first and the last latent coordinates
    n_latent_coords_per_column = max_displayed_latent_coords // 2
    # - - - stats on all metrics - - -
    data_flat = dict()
    outlier_limits = dict()  # measured lower and upper outliers bounds
    x_label = None  # Might include the number of not-displayed latent items
    for i, k in enumerate(metrics_names):
        latent_data = latent_metric.get_z(k)  # Not limited yet, but will be limited at the end of this iteration
        dim_z = latent_data.shape[1]
        data_flat[k] = latent_data.flatten()
        # Increased outliers display (IQR factor default value was 1.5) to be able to observe posterior collapse
        IQR_factor = max(1.5, 2.0 * np.log(dim_z) - 5.0)
        outlier_limits[k] = utils.stat.get_outliers_bounds(data_flat[k], IQR_factor=IQR_factor)
        outlier_limits[k] = (outlier_limits[k][0] - eps, outlier_limits[k][1] + eps)
        # get a subset if too large - otherwise the figured is rendered to 160MB (!!!) SVG files for dim_z = 2000
        data_flat[k] = utils.stat.get_random_subset_keep_minmax(data_flat[k], latent_data.shape[0] * 10)

        # FIXME build 2 tables of data_limited
        if i == 0:  # Build indices when processing the very first metric
            n_latent_coords_per_column = min(n_latent_coords_per_column, dim_z)  # Small dim_z: identical subplots...
            data_limited_indices = [range(0, n_latent_coords_per_column),
                                    range(dim_z-n_latent_coords_per_column, dim_z)]
            x_labels = ['First z coords ({} to {})'.format(data_limited_indices[0].start, data_limited_indices[0].stop-1),
                        'Last z coords ({} to {})'.format(data_limited_indices[1].start, data_limited_indices[1].stop-1)]
        for ii, data_limited_coord_range in enumerate(data_limited_indices):
            data_limited[ii][k] = latent_data[:, data_limited_coord_range]
    # - - - box plots (general and per component) - - -
    general_plots_eq_num_items = 10  # equivalent number of "small component boxplots", to properly divide fig width
    if figsize is None:
        figsize = (__param_width * (max_displayed_latent_coords + 8 + general_plots_eq_num_items), 6.5)
    fig, axes = plt.subplots(3, 3, figsize=figsize, sharex='col',
                             gridspec_kw={'width_ratios': [general_plots_eq_num_items,
                                                           n_latent_coords_per_column + 4,
                                                           n_latent_coords_per_column + 4]})
    flierprops = dict(marker='.', markerfacecolor='k', markersize=0.5, markeredgecolor='none')
    for i, k in enumerate(metrics_names):
        axes[i][0].boxplot(x=data_flat[k], vert=True, sym='.k', flierprops=flierprops)
        sns.boxplot(data=data_limited[0][k], ax=axes[i][1], fliersize=0.3, linewidth=0.5)
        sns.boxplot(data=data_limited[1][k], ax=axes[i][2], fliersize=0.3, linewidth=0.5)
    # - - - axes labels, limits and ticks - - -
    axes[2][0].set_xticks((1.0, ))
    axes[2][0].set_xticklabels(('all (rnd subset)', ))
    axes[0][0].set(ylabel='$q_{\phi}(z_0|x) : \mu_0$')
    axes[1][0].set(ylabel='$q_{\phi}(z_0|x) : \sigma_0$')
    axes[2][0].set(ylabel='$z_K$ samples')
    # mu0 and sigma0: unknown distributions, limit detailed per-component display to exclude outliers
    # zK (supposed to be approx. Standard Gaussian for training data):
    outlier_limits['sigma'] = (0.0, 1.0)  # sigma: use fixed scale (we know what sigma should look like...)
    for i in [1, 2]:
        axes[0][i].set(ylim=outlier_limits['mu'])
        axes[1][i].set(ylim=outlier_limits['sigma'])
        axes[2][i].set_xticklabels([str(t) for t in data_limited_indices[i-1]])
    #     limit display to +/- 4 std (-4std: cumulative distributions < e-4)
    axes[2][1].set(xlabel=x_labels[0], ylim=[-4.0, 4.0])
    axes[2][2].set(xlabel=x_labels[1], ylim=[-4.0, 4.0])
    # Target 25 and 75 percentiles as horizontal lines (+/- 0.6745 for standard normal distributions)
    for i in [1, 2]:
        axes[2][i].hlines([-0.6745, 0.0, 0.6745], -0.5, n_latent_coords_per_column-0.5, colors='grey', linewidth=0.5)
    # TODO x ticks axes 2
    # Small ticks for right subplots only (component indexes)
    for ax in [axes[2][1], axes[2][2]]:
        for tick in ax.get_xticklabels():
            tick.set_rotation(90)
            tick.set_fontsize(8)
    fig.tight_layout()
    return fig, axes


def plot_spearman_correlation(latent_metric: logs.metrics.LatentMetric):
    """ Plots the spearman correlation matrix (full, and with zeroed diagonal)
    and returns fig, axes """
    # http://jkimmel.net/disentangling_a_latent_space/ : Uncorrelated (independent) latent variables are necessary
    # but not sufficient to ensure disentanglement...
    corr = latent_metric.get_spearman_corr('zK')
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    im = axes[0].matshow(corr, cmap='viridis', vmin=-1.0, vmax=1.0)
    clb = fig.colorbar(im, ax=axes[0], orientation='vertical')
    axes[0].set_xlabel('zK Spearman corr')
    # 0.0 on diagonal - to get a better view on variations (2nd plot)
    corr = latent_metric.get_spearman_corr_zerodiag('zK')
    max_v = np.abs(corr).max()
    im = axes[1].matshow(corr, cmap='viridis', vmin=-max_v, vmax=max_v)
    clb = fig.colorbar(im, ax=axes[1], orientation='vertical')
    axes[1].set_xlabel('zeroed diagonal')
    for ax in axes:
        for tick in ax.get_xticklabels():
            tick.set_rotation(90)
    fig.tight_layout()
    return fig, axes


def plot_network_parameters(network_params: Dict[str, Dict[str, np.ndarray]]):
    n_rows = 2 * len(network_params)
    if n_rows == 0:
        raise AssertionError("Network params should contain at least 1 key.")
    param_names = None
    for k, layer_params in network_params.items():
        _param_names = list(layer_params.keys())
        if param_names is not None and param_names != _param_names:
            raise AssertionError("All network layers should contain the same parameters' names (found: {} and {}), "
                                 "using the same order.".format(_param_names, param_names))
        param_names = _param_names
    # 2 rows / layer: with and without outliers
    fig, axes = plt.subplots(n_rows, len(param_names), figsize=(len(param_names) * 2, n_rows * 2))
    layer_idx = 0
    for layer_name, layer_params in network_params.items():
        param_idx = 0
        for param_name, param_values in layer_params.items():
            color = 'C{}'.format(layer_idx % 10)
            sns.violinplot(x=param_values, ax=axes[layer_idx*2, param_idx], color=color)
            sns.violinplot(x=utils.stat.remove_outliers(param_values), ax=axes[layer_idx*2 + 1, param_idx], color=color)
            axes[layer_idx * 2, param_idx].grid(axis='x', linestyle='-')
            axes[layer_idx * 2 + 1, param_idx].grid(axis='x', linestyle='-')
            if layer_idx == 0:
                axes[layer_idx * 2, param_idx].set_title(param_name)
            param_idx += 1
        axes[layer_idx * 2, 0].set_ylabel('{} - raw data'.format(layer_name))
        axes[layer_idx * 2 + 1, 0].set_ylabel('{} - no outliers'.format(layer_name))
        layer_idx += 1
    fig.tight_layout()
    return fig, axes


def _configure_params_plot_x_tick_labels(ax):
    for tick in ax.get_xticklabels():
        tick.set_rotation(90)
        tick.set_fontsize(__x_tick_font_size)


def plot_synth_preset_param(ref_preset: Sequence, inferred_preset: Optional[Sequence] = None,
                            preset_UID=None, dataset: Optional[PresetDataset] = None):  # TODO figsize arg
    """ Plots reference parameters values of 1 preset (full VSTi-compatible representation),
    and their corresponding reconstructed values if given.

    :param ref_preset: A ground-truth preset (must be full) - 1d list of parameters values
    :param inferred_preset: Reconstructed preset (optional) - 1d list of parameters values
    :param dataset: (optional) PresetDataset class, to improve the display (param names, cardinality, ...)
    """
    if inferred_preset is not None:
        assert len(ref_preset) == len(inferred_preset)
    fig, ax = plt.subplots(1, 1, figsize=(__param_width * len(ref_preset), 4))  # TODO dynamic fig size
    # Params cardinality: deduced from the synth arg (str)
    if dataset is not None:
        if dataset.synth_name.lower() == 'dexed':
            # Gets cardinality of *all* params (including non-learnable)
            # Directly ask for quantized values
            params_quant_values = [dataset.get_preset_param_quantized_steps(i, learnable_representation=False)
                                   for i in range(len(ref_preset))]
            for i, y_values in enumerate(params_quant_values):
                if y_values is not None and len(y_values) < 50:  # Discrete param with small cardinality
                    marker = '_' if y_values.shape[0] > 1 else 'x'
                    sns.scatterplot(x=[i for _ in range(y_values.shape[0])], y=y_values, marker=marker,
                                    color='grey', ax=ax)
        else:
            raise NotImplementedError("Synth '{}' parameters cannot be displayed".format(dataset.synth_name))
    # For easier seaborn-based plot: we use a pandas dataframe
    df = pd.DataFrame({'param_idx': range(len(ref_preset)), 'ref_preset': ref_preset})
    learnable_param_indexes = dataset.learnable_params_idx if dataset is not None else None
    if learnable_param_indexes is not None:
        df['is_learnable'] = [(idx in learnable_param_indexes) for idx in range(len(ref_preset))]
    else:
        df['is_learnable'] = [True for idx in range(len(ref_preset))]
    # Scatter plot for "faders" values
    sns.scatterplot(data=df, x='param_idx', y='ref_preset', ax=ax,
                    hue="is_learnable",
                    palette=("blend:#BBB,#06D" if learnable_param_indexes is not None else "deep"))
    if inferred_preset is not None:
        df['inferred_preset'] = inferred_preset
        sns.scatterplot(data=df, x='param_idx', y='inferred_preset', ax=ax,
                        hue="is_learnable",
                        palette=("blend:#BBB,#D60" if learnable_param_indexes is not None else "husl"))
    ax.set_xticks(range(len(ref_preset)))
    param_names = dataset.preset_param_names if dataset is not None else None
    ax.set_xticklabels(['{}.{}'.format(idx, ('' if param_names is None else param_names[idx]))
                             for idx in range(len(ref_preset))])
    ax.set(xlabel='', ylabel='Param. value', xlim=[0-0.5, len(ref_preset)-0.5])
    ax.get_legend().remove()
    if preset_UID is not None:
        ax.set_title("Preset UID={} (VSTi numerical parameters)".format(preset_UID))
    # vertical "faders" separator lines
    plt.vlines(x=np.arange(len(ref_preset) + 1) - 0.5, ymin=0.0, ymax=1.0, colors='k', linewidth=1.0)
    _configure_params_plot_x_tick_labels(ax)
    fig.tight_layout()
    return fig, ax


def _get_learnable_preset_xticklabels(idx_helper: PresetIndexesHelper):
    vst_param_names = idx_helper.vst_param_names
    x_tick_labels = list()
    # Param names - vst-param by vst-param. We do not actually care about the learnable index
    for vst_idx, learnable_indexes in enumerate(idx_helper.full_to_learnable):
        if learnable_indexes is not None:  # learnable only
            if isinstance(learnable_indexes, Iterable):  # cat learnable representation
                for i, _ in enumerate(learnable_indexes):
                    if i == 0:
                        x_tick_labels.append('{}.{}.{}'.format(vst_idx, i, vst_param_names[vst_idx]))
                    else:
                        x_tick_labels.append('{}.{}'.format(vst_idx, i))
            else:  # numerical learnable representation
                x_tick_labels.append('{}.{}'.format(vst_idx, vst_param_names[vst_idx]))
    return x_tick_labels


def plot_synth_learnable_preset(learnable_preset, idx_helper: PresetIndexesHelper, preset_UID=None, figsize=None):
    """ Plots a single learnable preset (provided as 1D Tensor) """
    n_params = learnable_preset.size(0)
    assert n_params == idx_helper.learnable_preset_size
    fig, ax = plt.subplots(1, 1, figsize=(__param_width * n_params, 4))  # TODO dynamic fig size
    learnable_param_indexes = range(idx_helper.learnable_preset_size)
    # quantized values - plot now
    params_quant_values = [idx_helper.get_learnable_param_quantized_steps(idx)
                           for idx in learnable_param_indexes]
    for i, y_values in enumerate(params_quant_values):
        if y_values is not None:  # Discrete param only
            marker = '_' if y_values.shape[0] > 1 else 'x'
            sns.scatterplot(x=[i for _ in range(y_values.shape[0])], y=y_values, marker=marker,
                            color='grey', ax=ax)
    # For easier seaborn-based plot: we use a pandas dataframe
    df = pd.DataFrame({'param_idx': range(n_params), 'ref_preset': learnable_preset})
    df['is_learnable'] = [True for _ in range(n_params)]
    # Scatter plot for "faders" values
    sns.scatterplot(data=df, x='param_idx', y='ref_preset', ax=ax)
    ax.set_xticks(range(n_params))
    ax.set_xticklabels(_get_learnable_preset_xticklabels(idx_helper))
    ax.set(xlabel='', ylabel='Param. value', xlim=[0-0.5, n_params-0.5])
    if preset_UID is not None:
        ax.set_title("Preset UID={} (learnable parameters)".format(preset_UID))
    # vertical "faders" separator lines
    plt.vlines(x=np.arange(n_params + 1) - 0.5, ymin=0.0, ymax=1.0, colors='k', linewidth=1.0)
    _configure_params_plot_x_tick_labels(ax)
    fig.tight_layout()
    return fig, ax


def plot_synth_preset_error(v_out: torch.Tensor, v_in: torch.Tensor, idx_helper: PresetIndexesHelper,
                            apply_softmax_to_v_out=False,
                            mae_y_limit=0.59, boxplots_y_limits=(-1.1, 1.1), figsize=None):
    """ Uses boxplots to show the error between inferred (out) and GT (in) preset parameters.
    Displays very large figures, because each logit has its own plot column.

    :param mae_y_limit: Constant y-axis upper display limit (to help visualize improvements during training).
        Won't be used is a computed MAE is actually greater than this value.
    :param boxplots_y_limits: Constant y-axis box plots display limits.
    :param idx_helper: to improve the display (param names, cardinality, ...) """
    # init
    warnings.warn("DEPRECATED - does not supports preset 2D", category=DeprecationWarning)
    if apply_softmax_to_v_out:
        raise NotImplementedError("'apply_softmax_to_v_out' as using the deprecated preset activation.... "
                                  "but activations are now handled by probability distributions themselves")
        act = None  # FIXME
        param_batch_errors = act(v_out) - v_in
    else:
        param_batch_errors = v_out - v_in
    n_params = param_batch_errors.size(1)
    assert n_params == idx_helper.learnable_preset_size
    batch_errors_np = param_batch_errors.numpy()
    if figsize is None:
        figsize = (__param_width * n_params, 5)
    # Search for synth groups of parameters
    param_groups_separations = []
    if idx_helper.synth_name.lower() == "dexed":
        groups_start_vst_indexes = [23 + 22*i for i in range(6)]  # 23. OP1 EG RATE 1 (1st operator MIDI param)
        cur_group = 0
        for learn_idx in range(n_params):
            # We add a new group when a threshold is reached
            if idx_helper.learnable_to_full[learn_idx] >= groups_start_vst_indexes[cur_group]:
                param_groups_separations.append(learn_idx - 0.5)
                cur_group += 1
            if cur_group >= 6:
                break  # Break for-loop after last groups separation
    else:
        print("[utils/figures.py] Unknown synth '{}' from given PresetIndexesHelper. "
              "No groups separations displayed on error plot.".format(idx_helper.synth_name))
    fig, axes = plt.subplots(2, 1, figsize=figsize, sharex=True)
    mae = np.abs(batch_errors_np).mean(axis=0)
    for learn_idx in range(batch_errors_np.shape[1]):
        learnable_model = idx_helper.vst_param_learnable_model[idx_helper.learnable_to_full[learn_idx]]
        if learnable_model == 'num':
            color = '#1F77B4'  # mpl C0 - blue
        elif learnable_model == 'cat':
            color = '#9467BD'  # mpl C4 - purple
        # Top axis: Mean Absolute Error
        axes[0].scatter(learn_idx, mae[learn_idx], color=color)
        # Bottom axis: box-plots
        axes[1].boxplot(x=batch_errors_np[:, learn_idx], positions=[learn_idx],
                        widths=0.8, flierprops={'marker': '.', 'markersize': 0.5},
                        boxprops={'color': color})
    axes[0].grid()
    axes[0].set(ylabel='MAE')
    y_max = max(mae.max()*1.02, mae_y_limit)  # dynamic limit
    axes[0].set_ylim([0.0, y_max])
    axes[0].set_title("Synth parameters inference error (blue/purple: numerical/categorical)")
    axes[1].grid(axis='y')
    axes[1].set(ylabel='Inference error')
    axes[1].set_ylim(boxplots_y_limits)
    axes[1].set_xticklabels(_get_learnable_preset_xticklabels(idx_helper))
    _configure_params_plot_x_tick_labels(axes[1])
    # Param groups separations lines
    if len(param_groups_separations) > 0:
        for row in range(len(axes)):
            axes[row].vlines(param_groups_separations, 0.0, 1.0,
                             transform=axes[row].get_xaxis_transform(), colors='C9', linewidth=1.0)
    fig.tight_layout()
    return fig, axes


def plot_synth_preset_vst_error(v_out: torch.Tensor, v_in: torch.Tensor, idx_helper: PresetIndexesHelper):
    raise NotImplementedError("Deprecated, does not supports Preset2D")
    #eval_criterion = model.loss.AccuracyAndQuantizedNumericalLoss(
    #    idx_helper, reduce_num_loss=False,reduce_accuracy=False, percentage_accuracy_output=False,
    #    compute_symmetrical_presets=True)
    eval_criterion = None
    accuracies, num_losses = eval_criterion(v_out, v_in)
    acc_vst_indices, num_vst_indices = list(accuracies.keys()), list(num_losses.keys())
    all_learned_vst_indices = acc_vst_indices + num_vst_indices

    # figsize correct
    fig, ax = plt.subplots(1, 1, figsize=(__param_width * len(all_learned_vst_indices), 3.5))
    for vst_idx, value in num_losses.items():
        ax.scatter(vst_idx, value, color='tab:blue')
    ax2 = ax
    for vst_idx, value in accuracies.items():
        ax2.scatter(vst_idx, 1.0 - value, color='tab:purple')
    ax.set_ylabel('MAE   [num. param]\n(1 - accuracy)   [cat. param]')
    # Lines to separate group
    if idx_helper.synth_name.lower() == "dexed":
        for x in [6, 7, 13, 14, 15, 19] \
                 + sum([[i + 22*op for i in [23, 27, 31, 32, 33, 36, 44]] for op in range(6)], []):
            ax.axvline(x-0.5, color='tab:gray', linewidth=0.5)
    else:
        warnings.warn("This plotting function is configured for the Dexed synth only.")

    # set axes limits
    ax.set_xlim([min(all_learned_vst_indices)-0.5, max(all_learned_vst_indices)+0.5])
    ax.set_ylim(bottom=0.0)

    # parameters' names
    ax.set_xticks(all_learned_vst_indices)
    ax.set_xticklabels(['{}.{}'.format(idx, idx_helper.vst_param_names[idx]) for idx in all_learned_vst_indices])
    _configure_params_plot_x_tick_labels(ax)

    fig.tight_layout()
    return fig, ax


def plot_preset2d_batch_error(v_out: torch.Tensor, v_in: torch.Tensor, idx_helper: Preset2dHelper,
                              log_scale=True):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        N = v_out.shape[0]
        # Retrieve numerical and categorical values
        v_out_numerical = v_out[:, idx_helper.matrix_numerical_bool_mask, 1]
        v_in_numerical = v_in[:, idx_helper.matrix_numerical_bool_mask, 1]
        v_error_numerical = v_out_numerical - v_in_numerical
        v_out_categorical = v_out[:, idx_helper.matrix_categorical_bool_mask, 0]
        v_in_categorical = v_in[:, idx_helper.matrix_categorical_bool_mask, 0]

        # TODO Create a proper subplot figure: numerical errrors boxplots, confusion matrix for each categorical param
        #    More width for cat param with a larger cardinal
        confusion_mat_axes_names = [['cat{}'.format(i)] * (1 + card // 4)  # Increase subplot Wsize if greater num of classes
                                    for i, card in enumerate(idx_helper.matrix_categorical_rows_card)]
        confusion_mat_axes_names = sum(confusion_mat_axes_names, [])
        fig, axes_dict = plt.subplot_mosaic(
            [['numerical'] * len(confusion_mat_axes_names),  # Numerical boxplots span over the whole width
             ['.'] * len(confusion_mat_axes_names),  # Empty row for numerical x tick labels
             confusion_mat_axes_names,  # 2 Rows for the confusion matrices
             confusion_mat_axes_names],
            figsize=(0.15 * idx_helper.n_learnable_numerical_params, 10.0), constrained_layout=True
        )
        # Numerical params: box plots
        numerical_error_df = pd.DataFrame(v_error_numerical, columns=idx_helper.matrix_numerical_params_names)
        # "wide-form" DataFrame (each numeric column will be plotted)
        sns.boxplot(data=numerical_error_df, fliersize=0.5, ax=axes_dict['numerical'])
        axes_dict['numerical'].tick_params(axis='x', rotation=90, labelsize='small')
        # Categorical params: confusion matrices
        confusion_mat_axes_names = ['cat{}'.format(i) for i in range(idx_helper.n_learnable_categorical_params)]
        for cat_idx, k in enumerate(confusion_mat_axes_names):
            ax = axes_dict[k]
            cmat = confusion_matrix(v_in_categorical[:, cat_idx], v_out_categorical[:, cat_idx],
                                    labels=list(range(idx_helper.matrix_categorical_rows_card[cat_idx])))
            cmat = np.log(1.0 + cmat) if log_scale else cmat
            sns.heatmap(cmat, ax=ax, cbar=False, square=True, cmap='viridis', vmin=0.0)
            # Axes: ticks and ticks labels
            class_indices = list(range(idx_helper.matrix_categorical_rows_card[cat_idx]))
            ax.set_xticks(np.asarray(class_indices) + 0.5)
            if len(class_indices) < 10:  # single-digit indices: force display all
                ax.set_xticklabels(class_indices)
            ax.tick_params(axis='x', labelsize='small')
            ax.set_yticks(np.asarray(class_indices) + 0.5)
            ax.yaxis.set_ticklabels([])
            # Title: keep a few letters only
            ax.set_title(idx_helper.matrix_categorical_params_names[cat_idx][0:(len(class_indices) + 1)])
    return fig, axes_dict


if __name__ == "__main__":

    # Latent Metric plot tests - filled with artificial values
    dim_z = 256
    latent_metrics = logs.metrics.LatentMetric(dim_z, 1000)  # dim_z is the 2nd dim in the hidden data member
    for k in ['mu', 'sigma', 'zK']:
        latent_metrics._z[k] = np.random.normal(0.0, 1.0, (1000, dim_z))
    latent_metrics._z['sigma'] = 1.0 / (1.0 + np.exp(latent_metrics._z['sigma']))
    latent_metrics.next_dataset_index = latent_metrics.dataset_len  # Force values, for testing only
    fig, ax = plot_latent_distributions_stats(latent_metrics)
    from pathlib import Path
    dir = Path(__file__).resolve().parent
    svg_file = dir.joinpath("plot_latent_distributions_stats_TEST.svg")
    fig.savefig(svg_file)
    print("SVG file size = {:.1f} M bytes".format(svg_file.stat().st_size / 1000000.0))
    plt.show()

    # Preset errors - from fake batches
    """
    import config
    _model_config, _train_config = config.ModelConfig(), config.TrainConfig()
    config.update_dynamic_config_params(_model_config, _train_config)
    import data.build
    _ds = data.build.get_dataset(_model_config, _train_config)
    fake_v_out, fake_v_in = list(), list()
    for _i in range(100):
        ref_preset = _ds[_i][1]
        fake_v_out.append(ref_preset)  # 1/3:   100% accuracy / 0% L1 error
        fake_v_out.append(_ds[_i+1000][1])
        #fake_v_out.append(_ds[_i+200][1])
        fake_v_in.append(ref_preset)
        fake_v_in.append(ref_preset)
        #fake_v_in.append(ref_preset)
    fake_v_out = torch.cat([torch.unsqueeze(v, dim=0) for v in fake_v_out])
    fake_v_in = torch.cat([torch.unsqueeze(v, dim=0) for v in fake_v_in])
    _fig, _ax = plot_preset2d_batch_error(fake_v_out, fake_v_in, _ds.preset_indexes_helper)
    plt.show()
    """

    a = 0


