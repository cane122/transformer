import torch

class LayerNormalization(torch.nn.Module):
    def __init__(self, d_model, device, epsilon=1e-5):
        super(LayerNormalization, self).__init__()
        self.epsilon = epsilon
        self.scale = torch.nn.Parameter(torch.ones(d_model))
        self.bias = torch.nn.Parameter(torch.zeros(d_model))

    def forward(self, x):
        # Get the device of the input tensor
        device = x.device

        # Move parameters to the device of the input tensor
        scale = self.scale.to(device)
        bias = self.bias.to(device)

        mean = torch.mean(x, dim=-1, keepdim=True)
        variance = torch.var(x, dim=-1, keepdim=True)
        normalized = (x - mean) / torch.sqrt(variance + self.epsilon)
        
        # Ensure that all tensors involved in the computation are on the same device
        output = scale * normalized + bias
        return output
