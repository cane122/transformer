import numpy as np
from multihead_attention import MultiHeadAttention
from layer_normalization import LayerNormalization
from positionwise_feed_forward import PositionwiseFeedForward

class DecoderLayer:
    def __init__(self, d_model, num_heads, d_ff):
        self.self_attention = MultiHeadAttention(d_model, num_heads)
        self.encoder_decoder_attention = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff)
        self.layer_norm1 = LayerNormalization(d_model)
        self.layer_norm2 = LayerNormalization(d_model)

    def forward(self, x, encoder_output):
        # Self-Attention Sublayer
        self_attention_output = self.self_attention(x, x, x)  # Q, K, V all come from x
        x = x + self_attention_output
        x = self.layer_norm1(x)

        # Encoder-Decoder Attention Sublayer
        encoder_decoder_attention_output = self.encoder_decoder_attention(x, encoder_output, encoder_output)
        x = x + encoder_decoder_attention_output
        x = self.layer_norm2(x)

        # Position-wise Feed-Forward Sublayer
        feed_forward_output = self.feed_forward(x)
        x = x + feed_forward_output
        return x
