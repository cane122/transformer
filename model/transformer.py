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
        self.num_layers = num_layers
        self.encoder = Encoder(num_layers, d_model, num_heads, d_ff, input_vocab_size, max_seq_length, drop_prob, device)
        self.decoder = Decoder(num_layers, d_model, num_heads, d_ff, target_vocab_size, max_seq_length, drop_prob, device)
        self.final_linear = nn.Sequential(
            PositionwiseFeedForward(d_model, d_ff, device),
            nn.Linear(d_model, target_vocab_size)
        )
        self.embedding = TokenEmbedding(input_vocab_size, d_model)
        self.device = device

    def forward(self, source, target):
        source = self.embedding(source)
        target = self.embedding(target)
        
        encoder_output = self.encoder.forward(source)
        decoder_output = self.decoder.forward(target, encoder_output)
        
        # Final linear layer now outputs correct vocabulary size
        output = self.final_linear(decoder_output)
        return output


    def generate_text(self, start_phrase, start_token, end_token, vocab, vocab_invers, max_length=50):
        """
        Generate text using the Transformer model with token indices.
        
        Args:
            start_phrase (str): Initial text to start generation
            start_token (int): Index of the start token
            end_token (int): Index of the end token
            vocab (dict): Dictionary mapping tokens to indices
            vocab_invers (dict): Dictionary mapping indices to tokens
            max_length (int): Maximum length of generated sequence
        
        Returns:
            str: Generated text
        """
        self.eval()  # Set model to evaluation mode
        
        # Initialize with start token and tokenize start phrase
        tokens = [start_token] + [vocab.get(token, vocab['<unk>']) 
                                for token in start_phrase.split()]
        
        input_tensor = torch.tensor([tokens], device=self.device)
        
        generated_indices = tokens[:]
        
        with torch.no_grad():
            for _ in range(max_length - len(tokens)):
                # Generate next token probabilities
                output = self.forward(input_tensor, input_tensor)
                
                # Get the most likely next token
                next_token_logits = output[:, -1, :]
                next_token = torch.argmax(next_token_logits, dim=-1).item()
                
                # Break if end token is generated
                if next_token == end_token:
                    break
                    
                # Add the predicted token to the sequence
                generated_indices.append(next_token)
                input_tensor = torch.cat([input_tensor, 
                                        torch.tensor([[next_token]], device=self.device)], 
                                    dim=1)
        
        # Convert indices back to tokens, excluding the start token
        generated_tokens = [vocab_invers[idx] for idx in generated_indices[1:]]
        
        # Join tokens to create final text
        return ' '.join(generated_tokens)