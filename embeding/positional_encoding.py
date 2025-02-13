import torch
import torch.nn as nn
import torch.nn.functional as F

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super().__init__()
        encoding = self.create_positional_encoding(d_model, max_len)
        self.register_buffer('encoding', encoding)  # Proper buffer registration

    def create_positional_encoding(self, d_model, max_len):
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(torch.log(torch.tensor(10000.0)) / d_model))
        encoding = torch.zeros((max_len, d_model))
        encoding.requires_grad = False

        encoding[:, 0::2] = torch.sin(position * div_term)
        encoding[:, 1::2] = torch.cos(position * div_term)

        #encoding = encoding.unsqueeze(0)  # Add batch dimension
        return encoding

    def forward(self, x):
        # self.encoding
        # [max_len = 512, d_model = 512]

        batch_size, seq_len, embedding_dim = x.size()
        # [batch_size = 64, seq_len = 50, embedding_dim = 128]

        max_seq_len = self.encoding.size(0)  # Get the maximum sequence length from encoding

        # Handle cases where the input sequence length exceeds the maximum sequence length in the encoding
        if seq_len > max_seq_len:
            # If the input sequence length is greater than the maximum sequence length in encoding,
            # trim the input sequence length to match the maximum sequence length in encoding
            seq_len = max_seq_len
            x = x[:, :seq_len, :]  # Trim input tensor to match the maximum sequence length
        
        # Add positional encoding to the input tensor
        positional_encoding = self.encoding[:seq_len, :].unsqueeze(0).expand(batch_size, -1, -1).to(x.device) 
        x_with_pos_encoding = x + positional_encoding

        return x_with_pos_encoding
