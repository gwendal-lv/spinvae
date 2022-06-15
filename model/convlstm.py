"""
File initially downloaded from https://github.com/ndrplz/ConvLSTM_pytorch/blob/master/convlstm.py

Modified in june 2022
"""

import torch.nn as nn
import torch


class ConvLSTMCell(nn.Module):
    def __init__(self, in_channels, hidden_channels, kernel_size, bias=True, peephole_connection=False):
        """
        Initialize ConvLSTM cell.

        Parameters
        ----------
        in_channels: int
            Number of channels of input tensor.
        hidden_channels: int
            Number of channels of hidden state.
        kernel_size: (int, int)
            Size of the convolutional kernel.
        bias: bool
            Whether to add the bias, or not.
        TODO Implement input dropout
        """
        super(ConvLSTMCell, self).__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.bias = bias
        self.peephole_connection = peephole_connection
        self.conv = nn.Conv2d(
            in_channels=self.in_channels + self.hidden_channels + (self.hidden_channels if peephole_connection else 0),
            out_channels=4 * self.hidden_channels,
            kernel_size=kernel_size,
            padding=(kernel_size[0] // 2, kernel_size[1] // 2),
            bias=self.bias
        )

    def forward(self, input_tensor, cur_state):
        """
        :param input_tensor: x<t> tensor
        :param cur_state: (h<t-1>, c<t-1>) tuple of tensors
        :return: (h<t>, c<t>) tuple of tensors
        """
        h_cur, c_cur = cur_state
        # concat h<t-1> and x<t> and compute all convs in a single operation
        if not self.peephole_connection:
            combined_conv_in = torch.cat([input_tensor, h_cur], dim=1)
        else:
            combined_conv_in = torch.cat([input_tensor, h_cur, c_cur], dim=1)
        combined_conv_out = self.conv(combined_conv_in)
        # Input (Update), Forget, Output gates
        i_f_o = torch.sigmoid(combined_conv_out[:, 0:3 * self.hidden_channels, :, :])
        i, f, o = torch.chunk(i_f_o, 3, dim=1)
        # New candidate cell memory
        c_tilde = torch.tanh(combined_conv_out[:, 3*self.hidden_channels:, :, :])

        c_next = f * c_cur + i * c_tilde
        h_next = o * torch.tanh(c_next)
        return h_next, c_next


class ConvLSTM(nn.Module):
    def __init__(self, in_channels, hidden_channels, kernel_size, num_layers, bias=True, bidirectional=False):
        """
        This class tries to remain close to torch.nn.LSTM but uses convolutions instead of matrix multiplications.
        Input and hidden states are image feature maps, and spatial structure is preserved through all
        LSTM layers.

        Batch dimension must always be first.

        Parameters:
            in_channels: Number of input channels
            hidden_channels: Number of hidden channels
            kernel_size: Size of kernel in convolutions
            num_layers: Number of LSTM layers stacked on each other
            bias: Bias or no bias in Convolution
        """
        super(ConvLSTM, self).__init__()
        self.bidirectional = bidirectional

        cell_list = []
        for i in range(num_layers):
            # Bidirectional: h outputs from both directions will be concatenated
            inner_input_ch = hidden_channels if not bidirectional else hidden_channels * 2
            cur_input_ch = in_channels if i == 0 else inner_input_ch
            cell_list.append(ConvLSTMCell(cur_input_ch, hidden_channels, kernel_size, bias=bias))
        self.cells = nn.ModuleList(cell_list)  # For cells to be registered as children modules

    @property
    def num_layers(self):
        return len(self.cells)

    def forward(self, input_tensor, initial_h_c_per_layer=None):
        """

        :param input_tensor: 5-D Tensor of shape (N, T, C_in, H, W)  (batch first)
        :param initial_h_c_per_layer: List of (h, c) tuples for each layer input. If this list's length is
            less than num_layers, the given (h, c) will be used for the first layers only.
        :returns: output_sequence, [layers_last_h, layers_last_c] where
            output_sequence is h<1>, ..., h<T> from the last layer
            and layers_last_h, layers_last_c are lists containing the h<T>, c<T> tensors from each layer.
        """
        N, seq_len, _, H, W = input_tensor.size()

        _initial_h_c = [(torch.zeros(N, cell.hidden_channels, H, W, device=input_tensor.device),
                         torch.zeros(N, cell.hidden_channels, H, W, device=input_tensor.device)) for cell in self.cells]
        if initial_h_c_per_layer is not None:
            for layer_idx in range(len(initial_h_c_per_layer)):
                _initial_h_c[layer_idx] = initial_h_c_per_layer[layer_idx]
        initial_h_c_per_layer = _initial_h_c

        last_states_per_layer = [[], []]  # The last hidden states h, c of each layer
        current_layer_input = input_tensor

        for layer_idx, cell in enumerate(self.cells):
            h, c = initial_h_c_per_layer[layer_idx]
            current_output_sequence = []  # Output sequence for a given layer
            for t in range(seq_len):
                h, c = cell(current_layer_input[:, t, :, :, :], (h, c))
                current_output_sequence.append(h)
            if self.bidirectional:
                for t in reversed(range(seq_len)):
                    h, c = cell(current_layer_input[:, t, :, :, :], (h, c))
                    current_output_sequence[t] = torch.cat((current_output_sequence[t], h), dim=1)  # Cat along ch dim
            # FIXME if bidirectional: need to cat the first h and c (as torch.nn.LSTM seems to do)
            last_states_per_layer[0].append(h)
            last_states_per_layer[1].append(c)
            current_output_sequence = torch.stack(current_output_sequence, dim=1)  # Stack - creates the sequence dim
            # Next inputs will be h<1>, ...., h<T-1> from this layer (or concatenated left/right h<t> if biLSTM)
            current_layer_input = current_output_sequence

        return current_output_sequence, last_states_per_layer


if __name__ == "__main__":
    import torchinfo

    # Fake sequence of image feature maps - batch 1st, seq index 2nd
    seq_len = 6
    n_in_ch = 32
    n_hid_ch = 10
    x_seq_in = torch.zeros((1, seq_len, n_in_ch, 17, 17))  # Hierarchical latent space: biggest expected size 17x17
    h_in, c_in = torch.ones((1, n_hid_ch, 17, 17)), torch.ones((1, n_hid_ch, 17, 17))
    # TODO tester cell seule
    _cell = ConvLSTMCell(n_in_ch, n_hid_ch, (1, 1))
    h_out, c_out = _cell(x_seq_in[:, 0, :, :, :], (h_in, c_in))

    # TODO tester multi-layer
    lstm = ConvLSTM(n_in_ch, n_hid_ch, (1, 1), 2, bidirectional=True)
    out = lstm(x_seq_in, [(h_in, c_in)])
    # TODO tester bidir multi-layer

    a = 0
    pass   # TODO
