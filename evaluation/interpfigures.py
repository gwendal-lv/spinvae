
from pathlib import Path
from typing import List, Optional

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from interpbase import InterpBase


def interp_results_boxplots(
        storage_paths: List[Path], models_names: Optional[List[str]] = None,
        metrics_to_plot=('smoothness', 'sum_squared_residuals'),
        exclude_min_max=True, exclude_features=('Noisiness', ),
        reference_model_idx=0
):
    """

    :param storage_paths:
    :param models_names:
    :param metrics_to_plot:
    :param exclude_min_max:
    :param exclude_features: e.g. Noisiness features seem badly estimated for the DX7 (because of the FM?). They
        are quite constant to 1.0 (absurd) and any slightly < 1.0 leads to diverging values after 1-std normalization
    :param reference_model_idx: The model to be considered as the reference for normalization of all metrics.
    :return:
    """
    # auto create model names if not given (use parent's name)
    if models_names is None:
        models_names = [p.parent.name for p in storage_paths]
    else:
        assert len(models_names) == len(storage_paths)
    # load data
    #    1st index: model index
    #    2nd index: metric type (e.g. smoothness, RSS, ...)
    #    3rd and 4th "dims": actual DataFrame whose index is an interp sequence index, columns are metrics' names
    models_interp_results = [InterpBase.get_interp_results(p) for p in storage_paths]

    # TODO remove unused columns (e.g. min / max)
    for metric_idx, metric_name in enumerate(metrics_to_plot):
        for model_idx, interp_results in enumerate(models_interp_results):
            results_df = interp_results[metric_name]
            cols = list(results_df.columns)
            for col in cols:
                try:
                    if exclude_min_max:
                        if col.endswith('_min') or col.endswith('_max'):
                            results_df.drop(columns=[col], inplace=True)  # must be inplace inside the for loop
                    for feat_name in exclude_features:
                        if col.startswith(feat_name):
                            results_df.drop(columns=[col], inplace=True)  # must be inplace inside the for loop
                except KeyError:
                    pass  # A feature might be removed multiple times (multiple overlapping removal criteria)

    # TODO for each feature, retrieve max of the 1st model, which can be used as a reference for normalization
    #   or retrieve mean?? (very different scales for different features)
    reference_results = models_interp_results[reference_model_idx]
    reference_norm_factors = {k: results_df.mean() for k, results_df in reference_results.items()}

    # Detailed boxplots: each metric has its own subplots
    fig, axes = plt.subplots(len(metrics_to_plot), 1, figsize=(12, len(metrics_to_plot) * 5))  # FIXME size
    if len(metrics_to_plot) == 1:
        axes = [axes]  # Add singleton dimension, for compatibility
    for metric_idx, metric_name in enumerate(metrics_to_plot):
        models_melted_results = list()
        for model_idx, interp_results in enumerate(models_interp_results):
            results_df = interp_results[metric_name]
            results_df = results_df / reference_norm_factors[metric_name]
            # https://stackoverflow.com/questions/49554139/boxplot-of-multiple-columns-of-a-pandas-dataframe-on-the-same-figure-seaborn
            melted_results_df = pd.melt(results_df)  # 2-cols DF: 'variable' (feature name) and 'value'
            melted_results_df['model_name'] = models_names[model_idx]
            models_melted_results.append(melted_results_df)
        models_melted_results = pd.concat(models_melted_results)
        # TODO use bright colors such that the black median line is easily visible
        sns.boxplot(data=models_melted_results, x="variable", y="value", hue="model_name",
                    ax=axes[metric_idx], showfliers=False, linewidth=0.75)  # TODO test linewidth 0.75
        if False:
            sns.pointplot(
                data=models_melted_results, x="variable", y="value", hue="model_name", ax=axes[metric_idx],
                errwidth=1.0, marker='.', scale=0.5, ci="sd", dodge=0.4, join=False,  # SD instead of 95% CI
                # palette="tab10", saturation=1.0  # useless
            )
        axes[metric_idx].set(title=metric_name)
        axes[metric_idx].tick_params(axis='x', labelrotation=90)
        #axes[metric_idx].set_yscale('log')
    fig.tight_layout()

    # TODO grouped boxplots

    return fig, axes


if __name__ == "__main__":
    # FIXME anonymize
    _storage_paths = [
        Path('/media/gwendal/Data/Interpolations/LinearNaive/interp_validation'),
        #Path('/media/gwendal/Data/Logs/preset-vae/presetAE/combined_vae_beta1.60e-04_presetfactor0.20/interp_validation'),
        Path('/media/gwendal/Data/Logs/preset-vae/presetAE/combined_vae_beta1.60e-04_presetfactor0.50/interp_validation')
    ]
    interp_results_boxplots(_storage_paths, ['LinearNaive', 'PresetAE0.5'])
    plt.show()

