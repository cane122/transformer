import numpy as np
from elements.decoder_layer import DecoderLayer
from embeding.token_embeding import TokenEmbedding
from embeding.positional_encoding import PositionalEncoding
from layers.positionwise_feed_forward import PositionwiseFeedForward

class Decoder:
    def __init__(self, num_layers, d_model, num_heads, d_ff, target_vocab_size, max_seq_length, drop_prob, device):
        self.decoder_layers = [DecoderLayer(d_model, num_heads, d_ff, drop_prob, device) for _ in range(num_layers)]
        self.embedding = TokenEmbedding(target_vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_seq_length)
        self.final_linear = PositionwiseFeedForward(d_model, target_vocab_size, device)

    def forward(self, target, encoder_output):
        # Apply token embedding and positional encoding
        x = self.embedding(target)
        x = self.positional_encoding(x)

        # Forward pass through each decoder layer
        for decoder_layer in self.decoder_layers:
            x = decoder_layer.forward(x, encoder_output)

        # Apply the final linear layer for output
        output = self.final_linear(x)

        return output.to(target.device)
