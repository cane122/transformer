import numpy as np
from transformer import Transformer
from encoder import Encoder
from decoder import Decoder
from positionwise_feed_forward import PositionwiseFeedForward

def main():
    # Define hyperparameters and model parameters
    num_layers = 4
    d_model = 256
    num_heads = 8
    d_ff = 1024
    input_vocab_size = 10000
    target_vocab_size = 10000
    max_seq_length = 100

    # Create an instance of the Transformer model
    transformer = Transformer(num_layers, d_model, num_heads, d_ff, input_vocab_size, target_vocab_size, max_seq_length)

    # Generate some example input data
    source = np.random.randn(64, max_seq_length, d_model)  # Example source data
    target = np.random.randn(64, max_seq_length, d_model)  # Example target data

    # Perform a forward pass through the Transformer
    output = transformer.forward(source, target)

    # Print the shape of the output (should be of shape [batch_size, max_seq_length, target_vocab_size])
    print("Output shape:", output.shape)

if __name__ == "__main__":
    main()
