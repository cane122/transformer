import torch

class MultiHeadAttention(torch.nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        assert d_model % num_heads == 0  # Ensure that d_model is divisible by num_heads
        self.depth = d_model // num_heads

        # Weight matrices for linear transformations
        self.wq = torch.nn.Parameter(torch.randn(d_model, d_model))
        self.wk = torch.nn.Parameter(torch.randn(d_model, d_model))
        self.wv = torch.nn.Parameter(torch.randn(d_model, d_model))

    def split_heads(self, x):
        # Reshape input to split into multiple heads
        x = x.view(x.size(0), -1, self.num_heads, self.depth)
        return x.permute(0, 2, 1, 3)

    def forward(self, q, k, v):
        # Linear transformations for queries, keys, and values
        q = torch.matmul(q, self.wq)
        k = torch.matmul(k, self.wk)
        v = torch.matmul(v, self.wv)

        # Split into multiple heads
        q = self.split_heads(q)
        k = self.split_heads(k)
        v = self.split_heads(v)

        # Scaled Dot-Product Attention
        scaled_scores = torch.matmul(q, k.permute(0, 1, 3, 2)) / torch.sqrt(torch.tensor(self.depth).float())
        attention_weights = torch.nn.functional.softmax(scaled_scores, dim=-1)
        attention_output = torch.matmul(attention_weights, v)

        # Combine multiple heads
        attention_output = attention_output.permute(0, 2, 1, 3)
        attention_output = attention_output.reshape(attention_output.size(0), -1, self.d_model)

        return attention_output