import torch
from torch.nn import functional as F
import torch.nn as nn
from embeding.token_embeding import TokenEmbedding
from layers.positionwise_feed_forward import PositionwiseFeedForward
from model.decoder import Decoder
from model.encoder import Encoder

class Transformer(nn.Module):
    def __init__(self, num_layers, d_model, num_heads, d_ff, input_vocab_size, target_vocab_size, max_seq_length, drop_prob, device):
        super(Transformer, self).__init__()
        # Initialize hyperparameters and create necessary components
        self.num_layers = num_layers
        self.encoder = Encoder(num_layers, d_model, num_heads, d_ff, input_vocab_size, max_seq_length, drop_prob,device)
        self.decoder = Decoder(num_layers, d_model, num_heads, d_ff, target_vocab_size, max_seq_length, drop_prob, device)
        self.final_linear = PositionwiseFeedForward(d_model, target_vocab_size, device)
        self.token_embedding = TokenEmbedding(input_vocab_size, d_model)
        self.device = device

    def forward(self, source, target):
        # Forward pass through the encoder
        encoder_output = self.encoder.forward(source)

        # Forward pass through the decoder
        decoder_output = self.decoder.forward(target, encoder_output)

        # Apply the final linear layer to obtain output probabilities
        output = self.final_linear.forward(decoder_output)

        return output


    def generate_text(self, start_token_index, end_token_index, vocabulary, max_length=50):
        """
        Generate text using the Transformer model.
        """
        input_tensor = torch.tensor([[start_token_index]], device=self.device)  # Initialize input tensor with start token index
        generated_text = [start_token_index]  # Initialize list to store generated tokens

        with torch.no_grad():  # Disable gradient tracking during inference
            for _ in range(max_length):
                output = self.forward(input_tensor, input_tensor)  # Forward pass to generate output probabilities

                next_token_index = output.argmax(dim=-1)[-1][-1].unsqueeze(0)  # Get index of the last token with maximum probability

                # Check if the next token index is within the vocabulary
                if next_token_index.item() not in vocabulary:
                    break  # End generation if the token index is not in the vocabulary
                

                generated_text.append(next_token_index.item())  # Append generated token to the list

                if next_token_index.item() == end_token_index:  # Compare with scalar value
                    break  # End generation if end token is generated

                input_tensor = torch.cat([input_tensor, next_token_index.unsqueeze(-1)], dim=-1)  # Concatenate new token to input tensor

        # Convert token indexes to text
        generated_text = [vocabulary[token_index] for token_index in generated_text]

        # Join the tokens to form the generated text
        generated_text = ' '.join(generated_text)

        return generated_text
