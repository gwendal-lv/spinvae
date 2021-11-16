"""
Functions to be used by logger.py using multiprocessing.
They are out of the class to prevent serialization and deadlock issues.
"""

import multiprocessing

import utils.figures


def get_stats_figures(epoch, super_metrics, networks_layers_params):
    figs_dict = {'LatentStats': utils.figures.plot_latent_distributions_stats(
        latent_metric=super_metrics['LatentMetric/Valid'])[0],
                 'LatentRhoCorr': utils.figures.plot_spearman_correlation(
                     latent_metric=super_metrics['LatentMetric/Valid'])[0]}
    for network_name, layers_params in networks_layers_params.items():  # key: e.g. 'Decoder'
        figs_dict['{}ParamsStats'.format(network_name)] = \
            utils.figures.plot_network_parameters(layers_params)[0]  # Retrieve fig only, not the axes
    return figs_dict


def get_stats_figs__multiproc(q: multiprocessing.Queue, epoch, super_metrics, networks_layers_params):
    q.put(get_stats_figures(epoch, super_metrics, networks_layers_params))

