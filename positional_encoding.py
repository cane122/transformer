import numpy as np

class PositionalEncoding:
    def __init__(self, d_model, max_seq_length):
        self.d_model = d_model
        self.max_seq_length = max_seq_length
        self.positional_encoding = self.generate_positional_encoding()

    def generate_positional_encoding(self):
        # Create the positional encoding matrix
        positional_encoding = np.zeros((self.max_seq_length, self.d_model))
        for pos in range(self.max_seq_length):
            for i in range(0, self.d_model, 2):
                positional_encoding[pos, i] = np.sin(pos / (10000 ** (i / self.d_model)))
                positional_encoding[pos, i + 1] = np.cos(pos / (10000 ** (i / self.d_model)))
        return positional_encoding

    def __call__(self, embeddings):
        # Add positional encodings to the input embeddings
        return embeddings + self.positional_encoding[:embeddings.shape[0]]

# Usage:
# positional_encoder = PositionalEncoding(d_model, max_seq_length)
# input_embeddings = token_embedding(input_tokens)
# input_with_position = positional_encoder(input_embeddings)
