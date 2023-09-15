import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict


def init_weights(m, mean: float = 0.0, std: float = 0.01):
    """
    Initialize parameters in convolutional and transpose convolutional layers (zero-mean Gaussian)
    Params:
        mean: mean
        std: standard deviation
    """
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        m.weight.data.normal_(mean, std)


def activation_func(type: str):
    """
    Return: activation function
    Params:
        type: 'relu', 'tanh', or 'leakyrelu'
    """
    if type.upper() == "RELU":
        nn_activation = nn.ReLU(inplace=False)
    elif type.upper() == "TANH":
        nn_activation = nn.Tanh()
    elif type.upper() == "LEAKYRELU":
        nn_activation = nn.LeakyReLU(0.1, inplace=False)
    else:
        raise ValueError("Unavailable activation type")
    return nn_activation


def build_MLP(dim_in: int, dense: list, activation_type: str, dropout_p: float = 0):
    """
    Return MLP and its feature dim of output
    Params:
        dim_in: input feature dim
        dense: list of layer dims
        activation_type: 'relu', 'tanh', or 'leakyrelu'
        dropout_p: dropout probability (not available for the output layer)
    """
    nn_activation = activation_func(activation_type)
    dic_layers = OrderedDict()
    if len(dense) == 0:
        dic_layers["Identity"] = nn.Identity()
        dim_y = dim_in
    else:
        for n in range(len(dense)):
            if n == 0:
                dic_layers["Linear" + str(n)] = nn.Linear(dim_in, dense[n])
            else:
                dic_layers["Linear" + str(n)] = nn.Linear(dense[n - 1], dense[n])
            if n != len(dense) - 1:
                dic_layers["activation" + str(n)] = nn_activation
                dic_layers["dropout" + str(n)] = nn.Dropout(p=dropout_p)
        dim_y = dense[-1]

    return nn.Sequential(dic_layers), dim_y


def build_conv2d(config: list, name: str, norm: bool = False):
    """
    Return: convolutional layer sequence
    Params:
        config: list of parameters, [input channel, output channel, kernel size, stride, padding]
        name: name of built CNN
        norm: use batch norm or not (default: no batch norm)
    """
    dic_layers = OrderedDict()
    [ch_in, ch_out, kernel_size, stride, padding] = config
    dic_layers[name] = nn.Conv2d(
        ch_in,
        ch_out,
        (kernel_size, kernel_size),
        (stride, stride),
        (padding, padding),
    )
    if norm:
        dic_layers[name + "_bn"] = torch.nn.BatchNorm2d(
            num_features=ch_out, affine=True, track_running_stats=True
        )
    return nn.Sequential(dic_layers)


class ResBlock2D(torch.nn.Module):
    """
    Resblock class in our work
    """

    def __init__(
        self, config: list, norm: bool = False, nblock: int = 1, dropout_p: float = 0
    ):
        """
        Params:
            config: list of parameters, [[input channel, output channel, kernel size, stride, padding]]
            norm: use batch norm or not (default: no batch norm)
            nblock: number of repeated blocks
            dropout_p: dropout probability (not available for the output layer)
        """
        super(ResBlock2D, self).__init__()
        self.c = config
        self.convs = nn.ModuleList([])
        self.n_layers = len(config)
        self.norm = norm
        for iblock in range(nblock):
            for n in range(self.n_layers):
                [ch_in, ch_out, kernel_size, stride, padding] = config[n]
                self.convs.append(
                    nn.Conv2d(
                        ch_in,
                        ch_out,
                        (kernel_size, kernel_size),
                        (stride, stride),
                        (padding, padding),
                    )
                )
        self.convs.apply(init_weights)
        if norm:
            self.bn = nn.ModuleList([])
            for iblock in range(nblock):
                [_, ch, _, _, _] = config[0]
                self.bn.append(
                    torch.nn.BatchNorm2d(
                        num_features=ch, affine=True, track_running_stats=True
                    )
                )
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, x):
        i = 0
        x_res = x
        for conv in self.convs:
            iblock = i // self.n_layers  # 当前是第几个block
            i = i + 1
            xt = F.leaky_relu(x, 0.1, inplace=False)
            x = conv(xt)
            x = self.dropout(x)

            if i % self.n_layers == 0:
                if self.norm:  # use batch normalization
                    bn = self.bn[iblock]
                    x = bn(x)
                x = x + x_res
                x_res = x
        return x


def build_GRU(dim_in: int, dim_hidden: int, num_layers: int, bidir: bool):
    """
    Return: GRU module
    Params:
        dim_in: input layer dim
        dim_hidden: hidden layer dim
        num_layers: number of layers
        bidir: bidirectional GRU or forward GRU
    """
    gru = nn.GRU(
        input_size=dim_in,
        hidden_size=dim_hidden,
        num_layers=num_layers,
        batch_first=True,
        bidirectional=bidir,
    )
    gru.flatten_parameters()
    return gru


def reparametrization(mean, logvar):
    """
    Return: sampled latent variables
    Params:
        mean: mean of latent variables
        logvar: log variance of latent variables
    """
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return torch.addcmul(mean, eps, std)
