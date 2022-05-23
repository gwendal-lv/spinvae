"""
Functions to be used by logger.py using multiprocessing.
They are out of the class to prevent serialization and deadlock issues.
"""

import multiprocessing

import utils.figures


def get_stats_figures(super_metrics, networks_layers_params):
    # TODO output values plot
    figs_dict = {
        'LatentStats/Valid': utils.figures.plot_latent_distributions_stats(
            latent_metric=super_metrics['LatentMetric/Valid'])[0],
        # 'LatentRhoCorr': utils.figures.plot_spearman_correlation(  # disabled, useless for big latent vectors
        #    latent_metric=super_metrics['LatentMetric/Valid'])[0]
        }
    try:
        fig, ax = utils.figures.plot_vector_distributions_stats(super_metrics, 'RegOutValues')
        figs_dict['RegOut'] = fig
    except AssertionError:  # The 'reg out' metrics will raise errors during pre-train (there were not filled)
        pass
    # Violin plots are too big to be computed and/or to be stored as tensorboard images... Disabled.
    #    But: tensorboard histograms are kind of OK (issue: scaling...)
    """
    for network_name, layers_params in networks_layers_params.items():  # key: e.g. 'Decoder'
        figs_dict['{}ParamsStats'.format(network_name)] = \
            utils.figures.plot_network_parameters(layers_params)[0]  # Retrieve fig only, not the axes
    """
    return figs_dict


def get_stats_figs__multiproc(q: multiprocessing.Queue, super_metrics, networks_layers_params):
    q.put(get_stats_figures(super_metrics, networks_layers_params))

