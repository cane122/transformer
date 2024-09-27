from torch import nn
import torch
from layers.multihead_attention import MultiHeadAttention
from layers.layer_normalization import LayerNormalization
from layers.positionwise_feed_forward import PositionwiseFeedForward

class DecoderLayer:
    def __init__(self, d_model, num_heads, d_ff, drop_prob, device):
        self.self_attention = MultiHeadAttention(d_model, num_heads, device)
        self.encoder_decoder_attention = MultiHeadAttention(d_model, num_heads, device)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, device)
        self.layer_norm1 = LayerNormalization(d_model, device)
        self.layer_norm2 = LayerNormalization(d_model, device)
        self.dropout1 = nn.Dropout(p=drop_prob)
        self.dropout2 = nn.Dropout(p=drop_prob)
        self.dropout3 = nn.Dropout(p=drop_prob)
        self.device = device

    def generate_causal_mask(self, seq_len):
        """Generates a causal (look-ahead) mask for self-attention."""
        mask = torch.triu(torch.ones(seq_len, seq_len, device=self.device), diagonal=1).bool()
        return mask  # Upper triangular matrix with 1s above diagonal (True means masked)

    def forward(self, x, encoder_output, encoder_decoder_mask=None):
        # Ensure that all tensors are on the same device
        x = x.to(self.device)
        encoder_output = encoder_output.to(self.device)

        # Self-Attention Sublayer with Causal Mask
        seq_len = x.size(1)
        self_attention_mask = self.generate_causal_mask(seq_len)  # Generate causal mask
        self_attention_output = self.self_attention(x, x, x, mask=self_attention_mask)
        self_attention_output = self_attention_output.to(self.device)
        x = x + self.dropout1(self_attention_output)
        x = self.layer_norm1(x)

        # Encoder-Decoder Attention Sublayer (optional mask can be passed for encoder-decoder attention)
        encoder_decoder_attention_output = self.encoder_decoder_attention(x, encoder_output, encoder_output, mask=encoder_decoder_mask)
        encoder_decoder_attention_output = encoder_decoder_attention_output.to(self.device)
        x = x + self.dropout2(encoder_decoder_attention_output)
        x = self.layer_norm2(x)

        # Position-wise Feed-Forward Sublayer
        feed_forward_output = self.feed_forward(x)
        feed_forward_output = feed_forward_output.to(self.device)
        x = x + self.dropout3(feed_forward_output)

        return x