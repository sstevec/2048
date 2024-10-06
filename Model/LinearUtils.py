from torch import nn
import torch.nn.init as init


def get_linear_layer(input_channels: int, output_channels: int, with_bias: bool = True) -> nn.Linear:
    layer = nn.Linear(in_features=input_channels, out_features=output_channels, bias=with_bias)
    nn.init.xavier_normal_(layer.weight)

    return layer
