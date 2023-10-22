import numpy as np
from multihead_attention import MultiHeadAttention
from positionwise_feed_forward import PositionwiseFeedForward

class EncoderLayer:
    def __init__(self, d_model, num_heads, d_ff):
        self.multihead_attention = MultiHeadAttention(d_model, num_heads)
        self.positionwise_feed_forward = PositionwiseFeedForward(d_model, d_ff)
        self.layer_norm1 = LayerNormalization(d_model)
        self.layer_norm2 = LayerNormalization(d_model)

    def forward(self, x):
        # Multi-Head Self-Attention
        attn_output = self.multihead_attention(x, x, x)

        # Residual Connection and Layer Normalization
        x = x + attn_output
        x = self.layer_norm1(x)

        # Position-wise Feed-Forward
        ff_output = self.positionwise_feed_forward(x)

        # Residual Connection and Layer Normalization
        x = x + ff_output
        x = self.layer_norm2(x)

        return x
