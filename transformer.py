import math_cane as math
from positionwise_feed_forward import PositionwiseFeedForward
from decoder import Decoder
from encoder import Encoder
import numpy as np

class Transformer:
    def __init__(self, num_layers, d_model, num_heads, d_ff, input_vocab_size, target_vocab_size, max_seq_length):
        # Initialize hyperparameters and create necessary components
        self.num_layers = num_layers
        self.encoder = Encoder(num_layers, d_model, num_heads, d_ff, input_vocab_size, max_seq_length)
        self.decoder = Decoder(num_layers, d_model, num_heads, d_ff, target_vocab_size, max_seq_length)
        self.final_linear = PositionwiseFeedForward(d_model, target_vocab_size)

    def forward(self, source, target):
        # Forward pass through the encoder
        encoder_output = self.encoder(source)

        # Forward pass through the decoder
        decoder_output = self.decoder(target, encoder_output)

        # Apply the final linear layer to obtain output probabilities
        output = self.final_linear(decoder_output)

        return output
