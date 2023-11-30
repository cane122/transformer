import numpy as np
from torch import nn
from layers.positionwise_feed_forward import PositionwiseFeedForward
from layers.layer_normalization import LayerNormalization
from layers.multihead_attention import MultiHeadAttention

class EncoderLayer:
    def __init__(self, d_model, num_heads, d_ff, drop_prob):
        self.multihead_attention = MultiHeadAttention(d_model, num_heads)
        self.positionwise_feed_forward = PositionwiseFeedForward(d_model, d_ff)
        self.layer_norm1 = LayerNormalization(d_model)
        self.layer_norm2 = LayerNormalization(d_model)
        self.droupout1 = nn.Dropout(p=drop_prob)
        self.droupout2 = nn.Dropout(p=drop_prob)
    def forward(self, x):
        # Multi-Head Self-Attention
        attn_output = self.multihead_attention.forward(x, x, x)

        # Residual Connection and Layer Normalization
        attn_output = self.droupout1(attn_output)
        x = x + attn_output
        x = self.layer_norm1.forward(x)

        # Position-wise Feed-Forward
        ff_output = self.positionwise_feed_forward.forward(x)

        # Residual Connection and Layer Normalization
        ff_output = self.droupout1(ff_output)
        x = x + ff_output
        x = self.layer_norm2.forward(x)

        return x
