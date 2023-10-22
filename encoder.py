import numpy as np
from encoder_layer import EncoderLayer

class Encoder:
    def __init__(self, num_layers, d_model, num_heads, d_ff, vocab_size, max_seq_length):
        self.encoder_layers = [EncoderLayer(d_model, num_heads, d_ff) for _ in range(num_layers)]
        self.embedding = TokenEmbedding(vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_seq_length)

    def forward(self, source):
        # Apply token embedding and positional encoding
        x = self.embedding(source)
        x = self.positional_encoding(x)

        # Forward pass through each encoder layer
        for encoder_layer in self.encoder_layers:
            x = encoder_layer(x)

        return x
