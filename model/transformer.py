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
        self.eval()
        # Start with only start token
        tokens = [start_token]
        
        # Convert start phrase to tokens (without special tokens)
        initial_tokens = [vocab.get(token, vocab['<unk>']) 
                        for token in start_phrase.split() 
                        if token not in ['<s>', '</s>']]
        
        tokens += initial_tokens
        input_tensor = torch.tensor([tokens], device=self.device)
        
        for _ in range(max_length - len(tokens)):
            output = self.forward(input_tensor, input_tensor)
            next_token = torch.argmax(output[0, -1, :]).item()
            
            if next_token == end_token:
                break
                
            tokens.append(next_token)
            input_tensor = torch.cat([
                input_tensor,
                torch.tensor([[next_token]], device=self.device)
            ], dim=1)
        
        # Filter out padding and special tokens
        filtered = [vocab_invers[idx] for idx in tokens 
                if idx not in [vocab['<pad>'], start_token, end_token]]
        
        return ' '.join(filtered)