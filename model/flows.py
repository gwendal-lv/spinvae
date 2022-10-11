"""
Normalizing flows classes and utils.
Most flow transform are directly defined in models constructors (see VAE.py, regression.py)
"""

import torch
from torch import nn
from torch.nn import functional as F

from nflows.nn import nets as nets
from nflows.flows.base import Flow
from nflows.distributions.normal import StandardNormal
from nflows.transforms.base import CompositeTransform
from nflows.transforms.coupling import AdditiveCouplingTransform, AffineCouplingTransform
from nflows.transforms.autoregressive import MaskedAffineAutoregressiveTransform
from nflows.transforms.permutations import ReversePermutation
from nflows.transforms.normalization import BatchNorm


def parse_flow_args(flow_arch: str, authorize_options=True):
    """ Parses flow arguments """
    flow_args = flow_arch.split('_')
    if len(flow_args) < 2:
        raise ValueError("flow_arch argument must contains at least a flow type and layers description, "
                         "e.g. 'realnvp_4l200'")
    if not authorize_options and len(flow_args) > 2:
        raise ValueError("Options unauthorized for this flow")
    flow_type = flow_args[0]
    flow_layers_args = flow_args[1].split('l')
    num_flow_layers = int(flow_layers_args[0])
    num_hidden_units_per_layer = int(flow_layers_args[1])
    # Optional arguments: default is False
    bn_between_layers, bn_inside_layers, output_bn = False, False, False
    for i in range(2, len(flow_args)):
        if flow_args[i].lower() == "BNbetween".lower():
            bn_between_layers = True
        elif flow_args[i].lower() == "BNinternal".lower() or flow_args[i].lower() == "BNinside".lower():
            bn_inside_layers = True
        elif flow_args[i].lower() == "outputBN".lower() or flow_args[i].lower() == "BNoutput".lower():
            output_bn = True
        else:
            raise ValueError("Unrecognized flow argument '_{}'".format(flow_args[i]))
    return flow_type, num_flow_layers, num_hidden_units_per_layer, bn_between_layers, bn_inside_layers, output_bn



class CustomRealNVP(CompositeTransform):
    """ A slightly modified version of the SimpleRealNVP from nflows,
     which is a CompositeTransform and not a full Flow with base distribution. """
    def __init__(self, features: int, hidden_features: int, num_layers: int, num_blocks_per_layer=2,
                 use_volume_preserving=False, activation=F.relu, dropout_probability=0.0,
                 bn_within_layers=False, bn_between_layers=False, output_bn=False):

        coupling_constructor = AdditiveCouplingTransform if use_volume_preserving else AffineCouplingTransform

        mask = torch.ones(features)
        mask[::2] = -1

        use_dropout = True  # Quick and dirty: 'global' variable, as seen by the create_resnet function

        def create_resnet(in_features, out_features):
            return nets.ResidualNet(
                in_features,
                out_features,
                hidden_features=hidden_features,
                num_blocks=num_blocks_per_layer,
                activation=activation,
                dropout_probability=dropout_probability if use_dropout else 0.0,
                use_batch_norm=bn_within_layers,
            )

        layers = []
        for l in range(num_layers):
            use_dropout = l < (num_layers-2)  # No dropout on the 2 last layers
            transform = coupling_constructor(
                mask=mask, transform_net_create_fn=create_resnet
            )
            layers.append(transform)
            mask *= -1  # Checkerboard masking inverse
            # Possibly No batch norm on the last 2 layers
            if bn_between_layers and (l < (num_layers - 2) or output_bn):
                layers.append(BatchNorm(features=features))

        super().__init__(layers)



class CustomMAAF(CompositeTransform):
    """ Slightly modified MaskedAffineAutoregressiveTransform from nflows, does not include a base distrib. """
    def __init__(self, features: int, hidden_features: int, num_layers: int, num_blocks_per_layer=2,
                 activation=F.relu, dropout_probability=0.0,
                 batch_norm_within_layers=False, batch_norm_between_layers=False):
        super.__init__()  # TODO pass layer sequence
        pass  # TODO



class InverseFlow(nn.Module):
    """
    CompositeTransform (nflows package) wrapper which reverses the flow .inverse and .forward methods.

    Useful when combined with DataParallel (which only provides .forward calls) or reverse the fast/slow
    properties of the forward/inverse calls of the original flow.
    """
    def __init__(self, flow: CompositeTransform):
        super().__init__()
        raise AssertionError("This class messes autograd graphs (or only pytorch summaries???) and will be removed")
        assert isinstance(flow, CompositeTransform)
        self.flow = flow

    def forward(self, z):
        return self.flow.inverse(z)

    def inverse(self, z):
        return self.flow(z)



if __name__ == "__main__":
    pass


