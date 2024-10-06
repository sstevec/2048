import torch
from torch import nn, einsum
import torch.nn.functional as func
import math
from LinearUtils import get_linear_layer


class BaseAttention(nn.Module):
    def __init__(self,
                 input_q_channel,
                 input_k_channel,
                 input_v_channel,
                 attend_qk_channels,
                 attend_v_channels,
                 final_output_channels,
                 device,
                 num_heads: int = 8,
                 dropout_prob: float = 0.0):
        super(BaseAttention, self).__init__()
        self.num_heads = num_heads
        self.dropout = nn.Dropout(dropout_prob)

        # linear layer input size
        self.input_q_channel = input_q_channel
        self.input_k_channel = input_k_channel
        self.input_v_channel = input_v_channel

        # Initialize channel sizes
        self.qk_channels = attend_qk_channels
        self.v_channels = attend_v_channels

        self.query_norm = nn.LayerNorm(input_q_channel)
        self.query_linear = get_linear_layer(input_q_channel, self.qk_channels)

        self.key_norm = nn.LayerNorm(input_k_channel)
        self.key_linear = get_linear_layer(input_k_channel, self.qk_channels)

        self.value_norm = nn.LayerNorm(input_v_channel)
        self.value_linear = get_linear_layer(input_v_channel, self.v_channels)

        # final output linear layer
        self.output_channels = final_output_channels
        self.attention_output_linear = get_linear_layer(self.v_channels, self.output_channels)

        assert self.qk_channels % self.num_heads == 0, "qk_channels must be divisible by num_heads"
        assert self.v_channels % self.num_heads == 0, "v_channels must be divisible by num_heads"

        # Channel sizes per head
        self.qk_channels_per_head = self.qk_channels // self.num_heads
        self.v_channels_per_head = self.v_channels // self.num_heads

        self.q, self.k, self.v = None, None, None
        self.attn_probs = None

        self.device = device

    def forward(self, inputs_q, inputs_kv, attention_mask=None):
        # Linear projections, with layer norm
        self.q = self.query_linear(self.query_norm(inputs_q))
        self.k = self.key_linear(self.key_norm(inputs_kv))
        self.v = self.value_linear(self.value_norm(inputs_kv))

        # Reshape for multi-head attention
        self.q = self.reshape_for_heads(self.q, self.qk_channels_per_head)
        self.k = self.reshape_for_heads(self.k, self.qk_channels_per_head)
        self.v = self.reshape_for_heads(self.v, self.v_channels_per_head)

        # Scaled dot-product attention
        scale = 1.0 / math.sqrt(self.qk_channels_per_head)
        attn_scores = self.compute_attention_score() * scale

        # mask upper triangle if needed
        if attention_mask is not None:
            attn_scores = attn_scores.masked_fill(attention_mask == 0, float('-inf'))

        self.attn_probs = func.softmax(attn_scores, dim=-1)
        self.attn_probs = self.dropout(self.attn_probs)
        result = self.compute_weighted_v()

        # Combine heads
        result = torch.reshape(result, result.shape[:-2] + (-1,))

        return self.attention_output_linear(result)

    def reshape_for_heads(self, x, channels_per_head):
        b, i, c = x.size()
        return torch.reshape(x, (b, i, self.num_heads, channels_per_head))

    def compute_attention_score(self):
        return einsum('bnhc,bmhc->bhnm', self.q, self.k)

    def compute_weighted_v(self):
        return einsum('bhnm,bmhd->bnhd', self.attn_probs, self.v)

