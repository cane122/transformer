import torch

class MultiHeadAttention(torch.nn.Module):
    def __init__(self, d_model, num_heads, device):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        assert d_model % num_heads == 0  # Ensure that d_model is divisible by num_heads
        self.depth = d_model // num_heads

        # Weight matrices for linear transformations
        self.wq = torch.nn.Parameter(torch.randn(d_model, d_model)).to(device)
        self.wk = torch.nn.Parameter(torch.randn(d_model, d_model)).to(device)
        self.wv = torch.nn.Parameter(torch.randn(d_model, d_model)).to(device)

        self.device = device

    def split_heads(self, x):
        # Reshape input to split into multiple heads
        x = x.view(x.size(0), -1, self.num_heads, self.depth)
        return x.permute(0, 2, 1, 3)

    def forward(self, q, k, v, mask=None):
        # Move inputs to the same device as the parameters
        q, k, v = q.to(self.device), k.to(self.device), v.to(self.device)

        # Linear transformations for queries, keys, and values
        q = torch.matmul(q, self.wq)
        k = torch.matmul(k, self.wk)
        v = torch.matmul(v, self.wv)

        # Split into multiple heads
        q, k, v = self.split_heads(q), self.split_heads(k), self.split_heads(v)

        # Scaled Dot-Product Attention
        scaled_scores = torch.matmul(q, k.permute(0, 1, 3, 2)) / torch.sqrt(torch.tensor(self.depth, device=self.device).float())

        # Apply the mask (optional)
        if mask is not None:
            scaled_scores = scaled_scores.masked_fill(mask == False, -8)  # Use -1e9 instead of -inf

        # For numerical stability: subtract max value from scaled_scores
        scaled_scores = scaled_scores - scaled_scores.max(dim=-1, keepdim=True)[0]

        # Softmax along the last axis
        attention_weights = torch.nn.functional.softmax(scaled_scores, dim=-1)

        # Apply attention weights to the values
        attention_output = torch.matmul(attention_weights, v)

        # Combine multiple heads
        attention_output = attention_output.permute(0, 2, 1, 3)
        attention_output = attention_output.reshape(attention_output.size(0), -1, self.d_model)

        return attention_output

