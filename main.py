import torch
from model.transformer import Transformer
from model.encoder import Encoder
from model.decoder import Decoder
from layers.positionwise_feed_forward import PositionwiseFeedForward

def main():
    # Define hyperparameters and model parameters
    num_layers = 2
    d_model = 128
    num_heads = 4
    d_ff = 256
    input_vocab_size = 273992
    target_vocab_size = 273992
    max_seq_length = 50
    drop_prob = 0.4

    # Create an instance of the Transformer model
    transformer = Transformer(num_layers, d_model, num_heads, d_ff, input_vocab_size, target_vocab_size, max_seq_length, drop_prob)

    # Generate some example input data
    source = torch.randn(64, max_seq_length, d_model)  # Example source data
    target = torch.randn(64, max_seq_length, d_model)  # Example target data

    # Define a loss function (e.g., CrossEntropyLoss for sequence-to-sequence tasks)
    loss_function = torch.nn.CrossEntropyLoss()

    # Perform a forward pass through the Transformer
    output = transformer.forward(source, target)

    # Assume you have ground truth data (replace it with your actual ground truth)
    ground_truth = torch.randint(target_vocab_size, (64, max_seq_length))  
    
    # Calculate the loss
    loss = loss_function(output.view(-1, target_vocab_size), ground_truth.view(-1))

    # Print the loss
    print("Loss:", loss.item())

    # Print the shape of the output (should be of shape [batch_size, max_seq_length, target_vocab_size])
    print("Output shape:", output.shape)

if __name__ == "__main__":
    main()
