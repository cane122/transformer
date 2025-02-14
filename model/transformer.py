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


    def generate_text(self, start_phrase, start_token_idx, end_token_idx, vocab, vocab_invers, max_length=50):
        self.eval()
        # Start with the start token (as an integer)
        tokens = [start_token_idx]
        
        # Convert start_phrase (a string) into a list of token indices
        initial_tokens = [vocab.get(token, vocab['<unk>']) for token in start_phrase.split()]
        tokens += initial_tokens
        
        # Create the input tensor from tokens (all are ints)
        input_tensor = torch.tensor([tokens], device=self.device)
        
        # This list will store generated token indices
        generated = []
        
        for _ in range(max_length - len(tokens)):
            # Use the model to get the next token logits (using teacher forcing with all but last token)
            output = self.forward(input_tensor, input_tensor[:, :-1])
            next_token = torch.argmax(output[0, -1, :]).item()
            
            # Stop if the end token is produced
            if next_token == end_token_idx:
                break
            
            # If it's not a special token, add to generated list
            if next_token not in [vocab['<pad>'], start_token_idx, end_token_idx]:
                generated.append(next_token)
            
            # Append the new token to input_tensor for further generation
            input_tensor = torch.cat([
                input_tensor,
                torch.tensor([[next_token]], device=self.device)
            ], dim=1)
        
        # Convert generated token indices to tokens (strings)
        # No need to filter again if you already skipped special tokens
        generated_tokens = [vocab_invers[idx] for idx in generated]
        
        return ' '.join(generated_tokens)
