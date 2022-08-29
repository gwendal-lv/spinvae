from typing import Optional

import numpy as np

import librosa



def compute_interpolation_metrics(audio, num_metric_frames: int):
    """ Computes and returns the dict of metrics.
        Metrics keys ending with '_frames' are 2D metrics ; other metrics are average values across all frames.
        First dim of all arrays is always the interpolation step.

        Deprecated: should use Timbral Toolbox (MATLAB, G. Peeters 2011, updated)
    """
    interpolation_metrics = dict()
    # first, compute librosa spectrograms
    sr = audio[0][1]
    n_fft = 2048
    n_samples = audio[0][0].shape[0]
    frame_len = n_samples // num_metric_frames
    specs = [np.abs(librosa.stft(a[0], n_fft=n_fft, hop_length=n_fft // 2)) for a in audio]  # Linear specs
    # metrics "mask" if RMS volume is lower than a threshold  TODO use fft hop length
    rms = np.asarray([librosa.feature.rms(y=a[0], frame_length=n_fft, hop_length=n_fft // 2)[0] for a in audio])
    interpolation_metrics['rms'] = 10.0 * np.log10(np.mean(rms, axis=1))
    rms_mask = 10.0 * np.log10(rms)  # same number of frames for all other features
    rms = 10.0 * np.log10(reduce_num_frames(rms, num_frames=2 * num_metric_frames))  # More RMS frames
    interpolation_metrics['rms_frames'] = rms
    # harmonic/percussive/residuals separation ; get average value for each (1 frame for each interp step)
    h_p_separated = [librosa.decompose.hpss(s, margin=(1.0, 3.0)) for s in specs]
    res = [np.abs(specs[i] - (hp[0] + hp[1])) for i, hp in enumerate(h_p_separated)]
    harm = np.asarray([20.0 * np.log10(hp[0] + 1e-7).mean() for hp in h_p_separated])
    perc = np.asarray([20.0 * np.log10(hp[1] + 1e-7).mean() for hp in h_p_separated])
    res = np.asarray([20.0 * np.log10(r + 1e-7).mean() for r in res])
    # hpss: Absolute values are much less interesting than ratios (log diffs)
    interpolation_metrics['harm_perc_diff'] = harm - perc
    interpolation_metrics['harm_residu_diff'] = harm - res
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
    interpolation_metrics['chroma_argmax_avg'] = np.asarray([c.mean() for c in chroma_argmax])
    interpolation_metrics['chroma_argmax_std'] = np.asarray([c.std() for c in chroma_argmax])
    # 1D spectral features: centroid, rolloff, etc... (use mask to compute average)
    spec_features = np.log(np.asarray([librosa.feature.spectral_centroid(S=s, sr=sr)[0] for s in specs]))
    interpolation_metrics = mask_rms_and_add_1D_spectral_metric(
        interpolation_metrics, spec_features, 'spec_centroid', rms_mask)
    spec_features = 20.0 * np.log10(np.asarray([librosa.feature.spectral_flatness(S=s)[0] for s in specs]))
    interpolation_metrics = mask_rms_and_add_1D_spectral_metric(
        interpolation_metrics, spec_features, 'spec_flatness')  # no mask for flatness
    spec_features = np.log(np.asarray([librosa.feature.spectral_rolloff(S=s, sr=sr)[0] for s in specs]))
    interpolation_metrics = mask_rms_and_add_1D_spectral_metric(
        interpolation_metrics, spec_features, 'spec_rolloff')  # no mask for roll-off either
    spec_features = np.log(np.asarray([librosa.feature.spectral_bandwidth(S=s, sr=sr)[0] for s in specs]))
    interpolation_metrics = mask_rms_and_add_1D_spectral_metric(
        interpolation_metrics, spec_features, 'spec_bandwidth')  # no mask for bandwidth
    # TODO spectral contrast and MFCCs: 2D features - will be processed later
    spec_features = np.asarray([librosa.feature.mfcc(y=a[0], sr=sr, n_mfcc=13, hop_length=n_fft // 2)
                                for a in audio])
    interpolation_metrics['mfcc_full'] = spec_features  # will be processed later
    spec_features = np.asarray([librosa.feature.spectral_contrast(S=s, sr=sr) for s in specs])
    interpolation_metrics['spec_contrast_full'] = spec_features  # will be processed later
    return interpolation_metrics


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

