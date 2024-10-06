from BaseMultiheadAttention import BaseAttention
from CustomFNN import CustomFNN
from torch import nn


class TransformerLayer(nn.Module):
    def __init__(self, input_size, device, attention_in_size=None, dense_layer_dim=None, num_heads=8, drop_rate=0.1):
        super(TransformerLayer, self).__init__()
        self.input_size = input_size
        self.device = device

        # output size of linear layer of qkv mapping
        self.attention_in_size = attention_in_size if attention_in_size else input_size
        self.num_heads = num_heads
        self.drop_rate = drop_rate

        # self attention so input for q,k,v are the same size
        self.self_attn = BaseAttention(input_size, input_size, input_size, self.attention_in_size,
                                       self.attention_in_size, input_size, device, num_heads, drop_rate)

        # the in and out need to match the input size
        self.dense_layer_dim = dense_layer_dim
        if self.dense_layer_dim is None:
            self.dense_layer_dim = [input_size, input_size * 4, input_size]

        self.ffn = CustomFNN(self.dense_layer_dim, device, drop_rate=drop_rate)
        self.norm1 = nn.LayerNorm(self.attention_out_size)
        self.norm2 = nn.LayerNorm(self.attention_out_size)
        self.dropout1 = nn.Dropout(drop_rate)

    def forward(self, x):
        # Apply self-attention layer
        attn_output = self.self_attn(x, x, x)
        attn_output = self.dropout1(attn_output)
        x = x + attn_output
        x = self.norm1(x)

        # Apply position-wise feedforward network
        ffn_output = self.ffn(x)
        # no explicit drop out layer here because we have drop out in fnn
        x = x + ffn_output
        x = self.norm2(x)

        return x
