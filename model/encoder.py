import numpy as np
from elements.encoder_layer import EncoderLayer
from embeding.token_embeding import TokenEmbedding
from embeding.positional_encoding import PositionalEncoding

class Encoder:
    def __init__(self, num_layers, d_model, num_heads, d_ff, vocab_size, max_seq_length, drop_prob):
        self.encoder_layers = [EncoderLayer(d_model, num_heads, d_ff, drop_prob) for _ in range(num_layers)]
        self.embedding = TokenEmbedding(vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_seq_length)

    def forward(self, source):
        # Apply token embedding and positional encoding
        x = self.embedding(source)
        x = self.positional_encoding(x)

        # Forward pass through each encoder layer
        for encoder_layer in self.encoder_layers:
            x = encoder_layer.forward(x)

        return x
