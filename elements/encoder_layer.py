import numpy as np
from torch import nn
from layers.positionwise_feed_forward import PositionwiseFeedForward
from layers.layer_normalization import LayerNormalization
from layers.multihead_attention import MultiHeadAttention

class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, drop_prob, device):
        super(EncoderLayer, self).__init__()
        self.multihead_attention = MultiHeadAttention(d_model, num_heads, device)
        self.positionwise_feed_forward = PositionwiseFeedForward(d_model, d_ff, device)
        self.layer_norm1 = LayerNormalization(d_model, device)
        self.layer_norm2 = LayerNormalization(d_model, device)
        self.dropout1 = nn.Dropout(p=drop_prob)
        self.dropout2 = nn.Dropout(p=drop_prob)
        self.device = device  # Store device for use in forward method

    def forward(self, x):
        # Multi-Head Self-Attention
        attn_output = self.multihead_attention(x, x, x)

        # Ensure attn_output is on the same device as x
        attn_output = attn_output.to(x.device)

        # Residual Connection and Layer Normalization
        attn_output = self.dropout1(attn_output)
        x = x + attn_output
        x = self.layer_norm1(x)

        # Position-wise Feed-Forward
        ff_output = self.positionwise_feed_forward(x)

        # Ensure ff_output is on the same device as x
        ff_output = ff_output.to(x.device)

        # Residual Connection and Layer Normalization
        ff_output = self.dropout2(ff_output)
        x = x + ff_output
        x = self.layer_norm2(x)

        return x