import torch
from torch.nn import functional as F
import torch.nn as nn
from layers.positionwise_feed_forward import PositionwiseFeedForward
from model.decoder import Decoder
from model.encoder import Encoder

class Transformer(nn.Module):
    def __init__(self, num_layers, d_model, num_heads, d_ff, input_vocab_size, target_vocab_size, max_seq_length, drop_prob):
        super(Transformer, self).__init__()
        # Initialize hyperparameters and create necessary components
        self.num_layers = num_layers
        self.encoder = Encoder(num_layers, d_model, num_heads, d_ff, input_vocab_size, max_seq_length, drop_prob)
        self.decoder = Decoder(num_layers, d_model, num_heads, d_ff, target_vocab_size, max_seq_length, drop_prob)
        self.final_linear = PositionwiseFeedForward(d_model, target_vocab_size)

    def forward(self, source, target):
        # Forward pass through the encoder
        encoder_output = self.encoder.forward(source)

        # Forward pass through the decoder
        decoder_output = self.decoder.forward(target, encoder_output)

        # Apply the final linear layer to obtain output probabilities
        output = self.final_linear.forward(decoder_output)

        return output


    def generate_text(self, start_token, end_token, max_length=50):
        # Initialize the input tensor with the start token
        input_tensor = torch.tensor([[start_token]])

        for _ in range(max_length):
            # Forward pass through the model
            with torch.no_grad():
                output_probs = self.forward(input_tensor, input_tensor)

            # Sample the next token from the output probabilities
            next_token = torch.multinomial(F.softmax(output_probs[0, -1, :], dim=-1), 1)

            # Append the next token to the input for the next iteration
            input_tensor = torch.cat([input_tensor, next_token.unsqueeze(0).unsqueeze(0)], dim=1)

            # If the generated token is an end token, stop generation
            if next_token == end_token:
                break

        # Convert the generated tensor to a list of tokens
        generated_text = input_tensor.squeeze().tolist()

        return generated_text
