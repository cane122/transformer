import torch
import torch.nn as nn

class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size, embed_size):
        super(TokenEmbedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.vocab_size = vocab_size

    def forward(self, x):
        device = next(self.embedding.parameters()).device
        x = x.to(device)
        # Embed tokens and reshape to (batch_size, sequence_length, embed_size)
        embedded_tokens = self.embedding(x)

        return embedded_tokens

