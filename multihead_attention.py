import numpy as np

class MultiHeadAttention:
    def __init__(self, d_model, num_heads):
        self.num_heads = num_heads
        self.d_model = d_model
        assert d_model % num_heads == 0  # Ensure that d_model is divisible by num_heads
        self.depth = d_model // num_heads

        # Weight matrices for linear transformations
        self.wq = np.random.randn(d_model, d_model)
        self.wk = np.random.randn(d_model, d_model)
        self.wv = np.random.randn(d_model, d_model)

    def split_heads(self, x):
        # Reshape input to split into multiple heads
        x = np.reshape(x, (x.shape[0], -1, self.num_heads, self.depth))
        return np.transpose(x, (0, 2, 1, 3))

    def forward(self, q, k, v):
        # Linear transformations for queries, keys, and values
        q = np.dot(q, self.wq)
        k = np.dot(k, self.wk)
        v = np.dot(v, self.wv)

        # Split into multiple heads
        q = self.split_heads(q)
        k = self.split_heads(k)
        v = self.split_heads(v)

        # Scaled Dot-Product Attention
        scaled_scores = np.matmul(q, np.transpose(k, (0, 1, 3, 2))) / np.sqrt(self.depth)
        attention_weights = np.nn.softmax(scaled_scores, axis=-1)
        attention_output = np.matmul(attention_weights, v)

        # Combine multiple heads
        attention_output = np.transpose(attention_output, (0, 2, 1, 3))
        attention_output = np.reshape(attention_output, (attention_output.shape[0], -1, self.d_model))

        return attention_output