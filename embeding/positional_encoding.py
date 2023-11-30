import torch
import torch.nn as nn
import torch.nn.functional as F

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super(PositionalEncoding, self).__init__()
        self.encoding = self.create_positional_encoding(d_model, max_len)

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

        # self.encoding
        # [max_len = 512, d_model = 512]

        batch_size, seq_len = x.size()
        # [batch_size = 128, seq_len = 30]

        return self.encoding[:seq_len, :]