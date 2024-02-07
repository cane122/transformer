from torch import nn
from layers.multihead_attention import MultiHeadAttention
from layers.layer_normalization import LayerNormalization
from layers.positionwise_feed_forward import PositionwiseFeedForward

class DecoderLayer:
    def __init__(self, d_model, num_heads, d_ff, drop_prob,device):
        self.self_attention = MultiHeadAttention(d_model, num_heads, device)
        self.encoder_decoder_attention = MultiHeadAttention(d_model, num_heads, device)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, device)
        self.layer_norm1 = LayerNormalization(d_model, device)
        self.layer_norm2 = LayerNormalization(d_model, device)
        self.dropout1 = nn.Dropout(p=drop_prob)
        self.dropout2 = nn.Dropout(p=drop_prob)
        self.dropout3 = nn.Dropout(p=drop_prob)
        self.device = device

    def forward(self, x, encoder_output):
        # Ensure that all tensors are on the same device
        x = x.to(self.device)
        encoder_output = encoder_output.to(self.device)

        # Self-Attention Sublayer
        self_attention_output = self.self_attention(x, x, x)  # Q, K, V all come from x
        self_attention_output = self_attention_output.to(self.device)
        x = x + self.dropout1(self_attention_output)
        x = self.layer_norm1(x)

        # Encoder-Decoder Attention Sublayer
        encoder_decoder_attention_output = self.encoder_decoder_attention(x, encoder_output, encoder_output)
        encoder_decoder_attention_output = encoder_decoder_attention_output.to(self.device)
        x = x + self.dropout2(encoder_decoder_attention_output)
        x = self.layer_norm2(x)

        # Position-wise Feed-Forward Sublayer
        feed_forward_output = self.feed_forward(x)
        feed_forward_output = feed_forward_output.to(self.device)
        x = x + self.dropout3(feed_forward_output)

        return x