
from pathlib import Path
from typing import List, Optional

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import warnings

from evalconfig import InterpEvalConfig
from evaluation.interpbase import InterpBase
import utils.stat


def interp_results_boxplots(
        storage_paths: List[Path], models_names: Optional[List[str]] = None,
        metrics_to_plot=('smoothness', 'nonlinearity'),
        eval_config: Optional[InterpEvalConfig] = None,
        reference_model_idx=0,
        display_wilcoxon_tests=False,
):
    """

    :param storage_paths:
    :param models_names:
    :param metrics_to_plot:
    :param exclude_min_max:
    :param exclude_features:
    :param eval_config: if given, will be used to exclude_min_max, exclude_features, ....
    :param reference_model_idx: The model to be considered as the reference for normalization of all metrics.
    :return:
    """
    # auto create model names if not given (use parent's name)
    if models_names is None:
        models_names = [p.parent.name + '/' + p.name for p in storage_paths]
    else:
        assert len(models_names) == len(storage_paths)
    # load data
    #    1st index: model index
    #    2nd index: metric type (e.g. smoothness, RSS, ...)
    #    3rd and 4th "dims": actual DataFrame whose index is an interp sequence index, columns are metrics' names
    models_interp_results = [InterpBase.get_interp_results(p, eval_config) for p in storage_paths]

    # for each feature, compute normalisation factors from the 1st model, to be used for all models
    #   mean "without outliers" gives the best boxplots
    reference_results = models_interp_results[reference_model_idx]
    reference_norm_factors = {k: utils.stat.means_without_outliers(results_df)
                              for k, results_df in reference_results.items()}

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
        # use bright colors (pastel palette) such that the black median line is easily visible
        sns.boxplot(data=models_melted_results, x="variable", y="value", hue="model_name",
                    ax=axes[metric_idx], showfliers=False, linewidth=1.0, palette="pastel")
        axes[metric_idx].set_ylim(ymin=0.0)
        '''
        sns.pointplot(
            data=models_melted_results, x="variable", y="value", hue="model_name", ax=axes[metric_idx],
            errwidth=1.0, marker='.', scale=0.5, ci="sd", dodge=0.4, join=False,  # SD instead of 95% CI
        )
        '''
        axes[metric_idx].set(title=metric_name)
        axes[metric_idx].tick_params(axis='x', labelrotation=90)
        # axes[metric_idx].set_yscale('log')
        # If 2 models only, we may perform the wilcoxon paired test
        if len(models_interp_results) == 2:
            if display_wilcoxon_tests:
                # p_value < 0.05 if model [1] has significantly LOWER values than model [0] (reference)
                p_values, has_improved = utils.stat.wilcoxon_test(
                    models_interp_results[0][metric_name], models_interp_results[1][metric_name])
                axes[metric_idx].set(title="{} - Wilcoxon test: {}/{} significantly improved features".format(
                    metric_name, np.count_nonzero(has_improved.values), len(has_improved)))
        else:
            if display_wilcoxon_tests:
                warnings.warn("The Wilcoxon test requires to provide only 2 models (reference and another)")
    fig.tight_layout()

    # TODO grouped boxplots - or maybe edit figures after they have been saved as PDF?

    return fig, axes


def plot_improvements_vs_ref(improvements_df: pd.DataFrame, hparams: Optional[List[str]] = None):
    measurements_to_plot = ['wilcoxon_improved_features', 'median_variation_vs_ref', 'mean_variation_vs_ref']

    # 1) Plot numeric general improvement
    # auto-adapt height
    n_models = len(set(improvements_df['model'].values))
    fig1, axes1 = plt.subplots(1, 3, figsize=(14, 1 + 0.25 * n_models), sharey=True)
    for j, measurement_name in enumerate(measurements_to_plot):
        sns.barplot(data=improvements_df, y="model", x=measurement_name, ax=axes1[j])
    fig1.tight_layout()

    # 2) plot improvements vs model/train/interp hparams - scatter plots only
    if hparams is not None and len(hparams) >= 1:
        fig2, axes2 = plt.subplots(len(hparams), 3, figsize=(4 * 3, 1 + 3 * len(hparams)), sharex='col', sharey='row')
        axes2 = np.expand_dims(axes2, axis=0) if len(hparams) == 1 else axes2
        for i, hparam in enumerate(hparams):
            for j, measurement_name in enumerate(measurements_to_plot):
                # draw the evolution of means using lines (as done in comet.ml)
                h_params_values = np.unique(improvements_df[hparam].values)  # float equality: works OK with hparams
                mean_per_hparam = [improvements_df.loc[improvements_df[hparam] == v][measurement_name].values.mean()
                                   for v in h_params_values]
                axes2[i, j].plot(mean_per_hparam, h_params_values, color='k')  # 'vertical' plot
                # Then draw the actual scatter plot
                sns.scatterplot(data=improvements_df, x=measurement_name, y=hparam, ax=axes2[i, j],
                                hue='model', legend=False)
        fig2.tight_layout()
    else:
        fig2, axes2 = None, None

    return fig1, axes1, fig2, axes2


if __name__ == "__main__":
    # use for debugging only
    from evalconfig import InterpEvalConfig

    if False:  # test BOXPLOT
        _base_path = Path(__file__).resolve().parent.parent.parent.joinpath("Data_SSD/Logs")
        _storage_paths = [
            _base_path.joinpath('RefInterp/LinearNaive/interp9_valid'),
            _base_path.joinpath('preset-vae/presetAE/combined_vae_beta1.60e-04_presetfactor0.20/interp9_valid_uLin_zLin')
        ]
        interp_results_boxplots(_storage_paths, eval_config=InterpEvalConfig(), display_wilcoxon_tests=True)

    _improvements_df = InterpBase.compute_interp_improvement_vs_ref(InterpEvalConfig())
    plot_improvements_vs_ref(
        _improvements_df,
        hparams=['trncfg__beta', 'trncfg__params_loss_compensation_factor', 'u_curve', 'z_curve']  # TODO try others
    )

    plt.show()

