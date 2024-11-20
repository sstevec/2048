import torch
from torch import nn


# TODO add parameter to select different activation function
class CustomFNN(nn.Module):
    def __init__(self, layer_dim_array, device, drop_rate=0.1, use_layer_norm=True):
        super(CustomFNN, self).__init__()
        if len(layer_dim_array) < 2:
            print("Error: need at least 2 element in layer_dim_array to form a layer")

        self.layers = nn.ModuleList()

        if drop_rate is not None and drop_rate > 0.0:
            self.drop_rate = drop_rate
            self.drop_out_layer = nn.Dropout(p=drop_rate)

        self.use_layer_norm = use_layer_norm
        self.norm_layers = nn.ModuleList()

        for i in range(len(layer_dim_array) - 1):
            layer = nn.Linear(layer_dim_array[i], layer_dim_array[i + 1])
            self.layers.append(layer)
            if use_layer_norm:
                norm_layer = nn.LayerNorm(layer_dim_array[i + 1])
                self.norm_layers.append(norm_layer)
        self.device = device

    def forward(self, x):
        for i in range(len(self.layers) - 1):
            # fc layer
            x = self.layers[i](x)

            # layer norm
            if self.use_layer_norm:
                x = self.norm_layers[i](x)

            # activation
            x = nn.ReLU()(x)

            # dropout every 3 layers and after the first layer
            if i % 3 == 0 and self.drop_out_layer:
                x = self.drop_out_layer(x)

        x = self.layers[-1](x)
        if self.use_layer_norm:
            x = self.norm_layers[-1](x)
        return x
