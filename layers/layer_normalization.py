import torch

class LayerNormalization(torch.nn.Module):
    def __init__(self, d_model, epsilon=1e-5):
        super(LayerNormalization, self).__init__()
        self.epsilon = epsilon
        self.scale = torch.nn.Parameter(torch.ones(d_model))
        self.bias = torch.nn.Parameter(torch.zeros(d_model))

    def forward(self, x):
        mean = torch.mean(x, dim=-1, keepdim=True)
        variance = torch.var(x, dim=-1, keepdim=True)
        normalized = (x - mean) / torch.sqrt(variance + self.epsilon)
        output = self.scale * normalized + self.bias
        return output