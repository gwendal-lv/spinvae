import pathlib
from typing import Optional

import numpy as np

import librosa



def compute_audio_features(audio, num_metric_frames: int):
    """ Computes and returns the dict of metrics.
        Metrics keys ending with '_frames' are 2D metrics ; other metrics are average values across all frames.
        First dim of all arrays is always the interpolation step.

        Deprecated: use peer-reviewed Timbral Toolbox (MATLAB, G. Peeters 2011, updated) instead
    """
    audio_features = dict()
    # first, compute librosa spectrograms
    sr = audio[0][1]
    n_fft = 2048
    n_samples = audio[0][0].shape[0]
    frame_len = n_samples // num_metric_frames
    specs = [np.abs(librosa.stft(a[0], n_fft=n_fft, hop_length=n_fft // 2)) for a in audio]  # Linear specs
    # metrics "mask" if RMS volume is lower than a threshold  TODO use fft hop length
    rms = np.asarray([librosa.feature.rms(y=a[0], frame_length=n_fft, hop_length=n_fft // 2)[0] for a in audio])
    audio_features['rms'] = 10.0 * np.log10(np.mean(rms, axis=1))
    rms_mask = 10.0 * np.log10(rms)  # same number of frames for all other features
    rms = 10.0 * np.log10(reduce_num_frames(rms, num_frames=2 * num_metric_frames))  # More RMS frames
    audio_features['rms_frames'] = rms
    # harmonic/percussive/residuals separation ; get average value for each (1 frame for each interp step)
    h_p_separated = [librosa.decompose.hpss(s, margin=(1.0, 3.0)) for s in specs]
    res = [np.abs(specs[i] - (hp[0] + hp[1])) for i, hp in enumerate(h_p_separated)]
    harm = np.asarray([20.0 * np.log10(hp[0] + 1e-7).mean() for hp in h_p_separated])
    perc = np.asarray([20.0 * np.log10(hp[1] + 1e-7).mean() for hp in h_p_separated])
    res = np.asarray([20.0 * np.log10(r + 1e-7).mean() for r in res])
    # hpss: Absolute values are much less interesting than ratios (log diffs)
    audio_features['harm_perc_diff'] = harm - perc
    audio_features['harm_residu_diff'] = harm - res
    # Chromagrams (will use their own Constant-Q Transform)
    chromagrams = [librosa.feature.chroma_cqt(y=a[0], sr=sr, hop_length=n_fft // 2) for a in audio]
    """
    chromagrams = [c / c.sum(axis=0)[np.newaxis, :] for c in chromagrams]  # convert to probabilities
    chroma_value_tiles = np.tile(np.arange(12), (rms_mask.shape[1], 1)).T
    chroma_values = [c * chroma_value_tiles for c in chromagrams]  # Weight of each chroma * its 'scalar' value
    avg_chroma_value_frames = [np.sum(c_v, axis=0) for c_v in chroma_values]
    avg_chroma_values = [avg_c[rms_mask[i, :] > -120.0].mean() for i, avg_c in enumerate(avg_chroma_value_frames)]
    self.interpolation_metrics['chroma_value'] = np.asarray(avg_chroma_values)
    # Compute the std for each pitch (across all 'non-zero-volume' frames) then average all stds
    chroma_std = [np.std(c[:, rms_mask[i]>-120.0], axis=1).mean() for i, c in enumerate(chromagrams)]
    self.interpolation_metrics['chroma_std'] = np.asarray(chroma_std)
    """
    # Getting the argmax and the corresponding std is probably a simpler/better usage of chromas
    chroma_argmax = [np.argmax(c, axis=0)[rms_mask[i] > -120.0] for i, c in enumerate(chromagrams)]
    audio_features['chroma_argmax_avg'] = np.asarray([c.mean() for c in chroma_argmax])
    audio_features['chroma_argmax_std'] = np.asarray([c.std() for c in chroma_argmax])
    # 1D spectral features: centroid, rolloff, etc... (use mask to compute average)
    spec_features = np.log(np.asarray([librosa.feature.spectral_centroid(S=s, sr=sr)[0] for s in specs]))
    audio_features = mask_rms_and_add_1D_spectral_metric(
        audio_features, spec_features, 'spec_centroid', rms_mask)
    spec_features = 20.0 * np.log10(np.asarray([librosa.feature.spectral_flatness(S=s)[0] for s in specs]))
    audio_features = mask_rms_and_add_1D_spectral_metric(
        audio_features, spec_features, 'spec_flatness')  # no mask for flatness
    spec_features = np.log(np.asarray([librosa.feature.spectral_rolloff(S=s, sr=sr)[0] for s in specs]))
    audio_features = mask_rms_and_add_1D_spectral_metric(
        audio_features, spec_features, 'spec_rolloff')  # no mask for roll-off either
    spec_features = np.log(np.asarray([librosa.feature.spectral_bandwidth(S=s, sr=sr)[0] for s in specs]))
    audio_features = mask_rms_and_add_1D_spectral_metric(
        audio_features, spec_features, 'spec_bandwidth')  # no mask for bandwidth
    # TODO spectral contrast and MFCCs: 2D features - will be processed later
    spec_features = np.asarray([librosa.feature.mfcc(y=a[0], sr=sr, n_mfcc=13, hop_length=n_fft // 2)
                                for a in audio])
    audio_features['mfcc_full'] = spec_features  # will be processed later
    spec_features = np.asarray([librosa.feature.spectral_contrast(S=s, sr=sr) for s in specs])
    audio_features['spec_contrast_full'] = spec_features  # will be processed later
    return audio_features


def mask_rms_and_add_1D_spectral_metric(
        interpolation_metrics, m: np.ndarray, name: str, rms: Optional[np.ndarray] = None):
    valid_m = [m[i, :][rms[i, :] > -120.0] for i in range(rms.shape[0])] if rms is not None else m
    interpolation_metrics[name + '_avg'] = np.asarray([_m.mean() for _m in valid_m])
    interpolation_metrics[name + '_std'] = np.asarray([_m.std() for _m in valid_m])
    return interpolation_metrics


def reduce_num_frames(m: np.ndarray, num_frames: int):
    """ Reduces (using averaging) the number of frames of a given metric (1D or 2D for a given interpolation step).

    :param m: metric provided as a 2D or 3D numpy array.
    """
    # First output frame will use a smaller number of input frames (usually short attack)
    if len(m.shape) == 1:
        raise AssertionError("Please provide a 2D or 3D metric (must contain several frames for each interp step)")
    avg_len = 1 + m.shape[1] // num_frames
    first_len = m.shape[1] - (num_frames-1) * avg_len
    m_out = np.zeros((m.shape[0], num_frames))
    if len(m.shape) == 2:
        m_out[:, 0] = np.mean(m[:, 0:first_len], axis=1)
        for i in range(1, num_frames):
            i_in = first_len + (i-1) * avg_len
            m_out[:, i] = np.mean(m[:, i_in:i_in+avg_len], axis=1)
    else:
        raise NotImplementedError()
    return m_out



# Deprecated code.... backed up here, temp
def _compute_metric_statistics(flat_metric: np.ndarray, metric_name: str):
    """
    :returns: FIXME DOC sum_squared_normalized_residuals, linregress_R2, pearson_r_squared
    """
    # TODO acceleration RMS (smoothness factor)
    # First, we perform linear regression on the output features
    linregressions = [scipy.stats.linregress(np.linspace(0.0, 1.0, num=y.shape[0]), y) for y in flat_metric]
    linregr_R2 = np.asarray([l.rvalue ** 2 for l in linregressions])
    linregr_pvalues = np.asarray([l.pvalue for l in linregressions])
    # target features (ideal linear increase/decrease)
    target_metric = np.linspace(flat_metric[:, 0], flat_metric[:, -1], num=flat_metric.shape[1]).T
    target_amplitudes = flat_metric.max(axis=1) - flat_metric.min(axis=1)
    mean_target_amplitude = target_amplitudes.mean()
    target_max_abs_values = np.abs(flat_metric).max(axis=1)
    target_relative_variation = target_amplitudes / target_max_abs_values
    # Sum of squared (normalized) residuals
    residuals = (flat_metric - target_metric) / mean_target_amplitude
    # r2 pearson correlation - might trigger warnings if an array is (too) constant. Dismiss p-values
    pearson_r = np.asarray(
        [scipy.stats.pearsonr(target_metric[i, :], flat_metric[i, :])[0] for i in range(target_metric.shape[0])])
    pearson_r = pearson_r[~np.isnan(pearson_r)]
    # max of abs derivative (compensated for step size, normalized vs. target amplitude)
    diffs = np.diff(flat_metric, axis=1) * flat_metric.shape[1] / mean_target_amplitude
    max_abs_diffs = np.abs(diffs.max(axis=1))  # Biggest delta for all steps, average for all sequences
    return {'metric': metric_name,
            'sum_squared_residuals': (residuals ** 2).mean(),
            'linregression_R2': linregr_R2.mean(),
            'target_pearson_r2': (pearson_r ** 2).mean(),  # Identical to R2 when there is no nan r value
            'max_abs_delta': max_abs_diffs.mean()
            }


def _compute_and_save_interpolation_features(storage_path: pathlib.Path):  # TODO feature stats from another model
    """ Reads the 'all_interp_metrics.pkl' file (pre-computed audio features at each interp step) and computes:
        - constrained affine regression values (start and end values are start/end feature values). They can't be used
          to compute a R2 coeff of determination, because some values -> -infty (when target is close to constant)
        - R2 coefficients for each regression curve fitted to interpolated features (not the GT ideal features)
    """
    with open(storage_path.joinpath('all_interp_metrics.pkl'), 'rb') as f:
        all_interp_metrics = pickle.load(f)
    # TODO pre-preprocessing of '_full' metrics: compute their avg and std as new keys
    # Compute various stats  for all metrics - store all in a dataframe
    results_list = list()
    for k, metric in all_interp_metrics.items():
        # sklearn r2 score (coefficient of determination) expects inputs shape: (n_samples, n_outputs)
        #    We can't use it with constrained regressions (from start to end point) because R2 values
        #    become arbitrarily negative for flat target data (zero-slope)
        # So: we'll use the usual rvalue from linear regression, where R2 (determination coeff) = rvalue^2 >= 0.0
        #    (pvalue > 0.05 indicates that we can't reject H0: "the slope is zero")
        # TODO also compute sum of errors compared to the "ideal start->end affine regression"
        if len(metric.shape) == 2:  # scalar metric (real value for each step of each interpolation sequence)
            results_list.append(_compute_metric_statistics(metric, k))
        elif len(
                metric.shape) == 3:  # vector metric (for each step of each sequence) e.g. RMS values for time frames
            if '_frames' not in k:
                raise AssertionError("All vector metrics should have the '_frames' suffix (found: '{}').".format(k))
            # A regression is computed on each time frame (considered independent)
            for frame_i in range(metric.shape[2]):
                results_list.append(_compute_metric_statistics(metric[:, :, frame_i], k + '{:02d}'.format(frame_i)))
        elif len(
                metric.shape) == 4:  # 2D metric (matrix for each step of each sequence) e.g. MFCCs, spectral contrast
            continue  # TODO multivariate correlation coeff ?
            # MFCC: orthogonal features (in the frequency axis)
        else:
            raise NotImplementedError()
    # Compute global averages (_frames excluded)
    results_df = pd.DataFrame(results_list)
    df_without_frames = results_df[~results_df['metric'].str.contains('_frame')]
    global_averages = {'metric': 'global_average'}
    for k in list(df_without_frames.columns):
        if k != 'metric':
            global_averages[k] = df_without_frames[k].values.mean()
    results_df.loc[len(results_df)] = global_averages  # loc can add a new row to a DataFrame
    # TODO output results and/or store
    return results_df
