import torch
from torch import nn
import torch.nn.functional as func


# TODO add parameter to select different activation function
class CustomFNN(nn.Module):
    def __init__(self, layer_dim_array, device, drop_rate=0.1, use_layer_norm=True):
        super(CustomFNN, self).__init__()
        if len(layer_dim_array) < 2:
            print("Error: need at least 2 element in layer_dim_array to form a layer")

        self.layers = nn.ModuleList()
        self.drop_rate = drop_rate

        self.use_layer_norm = use_layer_norm
        self.norm_layers = nn.ModuleList()

        for i in range(len(layer_dim_array) - 1):
            layer = nn.Linear(layer_dim_array[i], layer_dim_array[i + 1])
            self.layers.append(layer)
            if use_layer_norm:
                norm_layer = nn.LayerNorm(layer_dim_array[i + 1])
                self.norm_layers.append(norm_layer)
        self.device = device

    def forward(self, x, is_training=True):
        for i in range(len(self.layers) - 1):
            x = self.layers[i](x)
            if self.use_layer_norm:
                x = self.norm_layers[i](x)
            x = torch.relu(x)

        last_x = self.layers[-1](x)
        if self.use_layer_norm:
            last_x = self.norm_layers[-1](last_x)
        if is_training and self.drop_rate is not None:
            last_x = func.dropout(last_x, p=self.drop_rate, training=is_training)
        return last_x
