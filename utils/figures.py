"""
Utilities for plotting various figures (spectrograms, ...)
"""
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

import logs.metrics

from data.abstractbasedataset import AudioDataset, PresetDataset
from data.preset import PresetIndexesHelper

import model.regression
import model.loss

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


def plot_train_spectrograms(x_in, x_out, sample_info, dataset: AudioDataset,
                            model_config, train_config):
    """ Wrapper for plot_spectrograms, which is made to be easily used during training/validation. """
    if dataset.multichannel_stacked_spectrograms:  # Plot only 1 preset ID, multiple midi notes
        midi_notes = model_config.midi_notes
        presets_UIDs = torch.ones((sample_info.shape[0],), dtype=sample_info.dtype, device=sample_info.device)
        presets_UIDs *= sample_info[0, 0]
        presets_names = [dataset.get_name_from_preset_UID(presets_UIDs[0].item())]  # we send a single name
        max_nb_specs = len(midi_notes)
    else:  # Plot multiple preset IDs, TODO possibly also different midi notes for different 1-ch spectrograms
        midi_notes = [(sample_info[i, 1].item(), sample_info[i, 2].item()) for i in range(sample_info.shape[0])]
        presets_UIDs = sample_info[:, 0]
        presets_names = list()
        for i in range(np.minimum(train_config.logged_samples_count, presets_UIDs.shape[0])):
            presets_names.append(dataset.get_name_from_preset_UID(presets_UIDs[i].item()))
        max_nb_specs = len(presets_names)
    return plot_spectrograms(x_in, x_out, presets_UIDs=presets_UIDs,
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


def plot_spectrograms_interp(u: np.ndarray, spectrograms: torch.Tensor,  # TODO allow MFCCs
                             z: Optional[torch.Tensor] = None, num_z_coords_to_show=40,
                             subplot_w_h=(1.5, 1.5), title: Optional[str] = None):
    """ Plots a "batch" of interpolated spectrograms. The number of spectrograms must be the same as the number of
    interpolation abscissae u. """
    if u.shape[0] != spectrograms.shape[0] or u.shape[0] != z.shape[0]:  # TODO all tests
        raise AssertionError("All input arrays/tensor must have the same length along dimension 0.")
    if u.shape[0] < 2:
        raise ValueError("This function requires 2 spectrograms or more")
    if spectrograms.shape[1] > 1:
        raise ValueError("This function supports single-channel spectrograms only.")
    n_rows = 2 + (1 if z is not None else 0)  # TODO add rows for other plots
    fig, axes = plt.subplots(n_rows, u.shape[0], sharey='row',
                             figsize=(u.shape[0]*subplot_w_h[0], n_rows*subplot_w_h[1] + (0 if title is None else 0.3)))
    # Data converted to numpy, compute deltas
    if spectrograms.shape[1] > 1:
        raise AssertionError("Input spectrograms must be single-channel.")
    specs_np = spectrograms[:, 0, :, :].clone().detach().cpu().numpy()
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
    delta_z_np = np.zeros_like(z_np) if z_np is not None else None
    if z_np is not None:
        for u_idx in range(1, u.shape[0]):  # First delta will remain zero
            delta_z_np[u_idx, :] = (z_np[u_idx, :] - z_np[u_idx-1, :]) / (u[u_idx] - u[u_idx-1])
    # Plots
    delta_specs_cbar_ax = None
    for u_idx in range(u.shape[0]):
        _u = u[u_idx]
        axes[0, u_idx].set_title(r'$u=$' + '{:.2f}'.format(_u))
        # Spectrograms
        im = librosa.display.specshow(specs_np[u_idx, :, :], shading='flat', ax=axes[0, u_idx], cmap='magma',
                                      vmin=specs_np.min(), vmax=specs_np.max())
        if u_idx > 0:
            max_abs_error = max(np.abs(delta_specs_np.min()), np.abs(delta_specs_np.max()))
            im = librosa.display.specshow(delta_specs_np[u_idx, :, :], shading='flat', ax=axes[1, u_idx], cmap='bwr',
                                          vmin=-max_abs_error, vmax=max_abs_error)
            if u_idx == 1:
                delta_specs_cbar_ax = fig.add_axes(axes[1, 0].get_position())
                clb = fig.colorbar(im, cax=delta_specs_cbar_ax)
        # Latent vectors
        if z_np is not None:
            colors = [mpl_colors.hsv_to_rgb([((c * 63.2) % 100)/100.0, 1.0, 0.85]) for c in z_indices_range]
            axes[2, u_idx].scatter(z_indices_range, z_np[u_idx, :], s=1, c=colors)
            axes[2, u_idx].grid(axis='y')
    # Display: labels, ...
    axes[0, 0].set_ylabel(r'Spectrogram $\mathbf{s}$')
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
    axes_to_reposition = [(delta_specs_cbar_ax, axes[1, 0])]  # TODO ajouter autres axes à repos (si non-None)
    for cb_ax, original_ax in axes_to_reposition:
        pos = original_ax.get_position().bounds  # (x0, y0, width, height)
        cb_ax.set_position([pos[0] + pos[2]*0.45, pos[1], pos[2]*0.1, pos[3]])
    return fig, axes


def plot_latent_distributions_stats(latent_metric: logs.metrics.LatentMetric, figsize=None, eps=1e-7,
                                    max_displayed_latent_coords=150):
    """ Uses boxplots to represent the distribution of the mu and/or sigma parameters of
    latent gaussian distributions. Also plots a general histogram of all samples.

    :param latent_metric:
    :param figsize:
    :param eps: Value added to bounds for numerical stability (when a latent value is constant).
    """
    metrics_names = ['mu', 'sigma', 'zK']
    data_limited = dict()
    # - - - stats on all metrics - - -
    data_full_flat = dict()
    outlier_limits = dict()  # measured lower and upper outliers bounds
    x_label = None  # Might include the number of not-displayed latent items
    for k in metrics_names:
        data_limited[k] = latent_metric.get_z(k)
        if k == 'sigma':  # log10 applied to sigma
            data_limited[k] = np.log10(data_limited[k])
        data_full_flat[k] = data_limited[k].flatten()
        if data_limited[k].shape[1] > max_displayed_latent_coords:
            x_label = 'z index (partial subplot: {} to {} are not displayed)'\
                .format(max_displayed_latent_coords, data_limited[k].shape[1]-1)
            data_limited[k] = data_limited[k][:, 0:max_displayed_latent_coords]
        outlier_limits[k] = utils.stat.get_outliers_bounds(data_full_flat[k])
        outlier_limits[k] = (outlier_limits[k][0] - eps, outlier_limits[k][1] + eps)
    # - - - box plots (general and per component) - - -
    general_plots_eq_num_items = 10  # equivalent number of "small component boxplots", to properly divide fig width
    if figsize is None:
        figsize = (__param_width * (data_limited['mu'].shape[1] + 8 + general_plots_eq_num_items), 6.5)
    fig, axes = plt.subplots(3, 2, figsize=figsize, sharex='col',
                             gridspec_kw={'width_ratios': [general_plots_eq_num_items, data_limited['mu'].shape[1] + 8]})
    flierprops = dict(marker='.', markerfacecolor='k', markersize=0.5, markeredgecolor='none')
    for i, k in enumerate(metrics_names):
        axes[i][0].boxplot(x=data_full_flat[k], vert=True, sym='.k', flierprops=flierprops)
        sns.boxplot(data=data_limited[k], ax=axes[i][1], fliersize=0.3, linewidth=0.5)
    # - - - axes labels, limits and ticks - - -
    axes[2][0].set_xticks((1.0, ))
    axes[2][0].set_xticklabels(('all', ))
    # mu0 and sigma0: unknown distributions, limit detailed per-component display to exclude outliers
    # zK (supposed to be approx. Standard Gaussian): limit display to +/- 4 std (-4std: cumulative distributions < e-4)
    axes[0][1].set(ylabel='$q_{\phi}(z_0|x) : \mu_0$', ylim=outlier_limits['mu'])
    axes[1][1].set(ylabel='$q_{\phi}(z_0|x) : \log_{10}(\sigma_0)$', ylim=outlier_limits['sigma'])
    axes[2][1].set(xlabel=('z index' if x_label is None else x_label), ylabel='$z_K$ samples', ylim=[-4.0, 4.0])
    # TODO Target 25 and 75 percentiles as horizontal lines (+/- 0.6745 for standard normal distributions)
    axes[2][1].hlines([-0.6745, 0.0, 0.6745], -0.5, data_limited['mu'].shape[1]-0.5, colors='grey', linewidth=0.5)
    #     And target outliers limits (Q1/Q3 +/- 1.5 IQR) as dotted lines
    # Small ticks for right subplots only (component indexes)
    for ax in [axes[0][1], axes[1][1], axes[2][1]]:
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
    if apply_softmax_to_v_out:
        act = model.regression.PresetActivation(idx_helper, cat_hardtanh_activation=False, cat_softmax_activation=True)
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
    eval_criterion = model.loss.AccuracyAndQuantizedNumericalLoss(
        idx_helper, reduce_num_loss=False,reduce_accuracy=False, percentage_accuracy_output=False,
        compute_symmetrical_presets=True)
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

