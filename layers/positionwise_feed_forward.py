import torch

class PositionwiseFeedForward(torch.nn.Module):
    def __init__(self, d_model, d_ff, device):
        super(PositionwiseFeedForward, self).__init__()
        self.device = device
        self.dense1 = torch.nn.Linear(d_model, d_ff).to(device)
        self.dense2 = torch.nn.Linear(d_ff, d_model).to(device)

    def forward(self, x):
        device = self.device
        # Apply the first fully connected layer and ReLU activation
        output = self.dense1(x.to(device))
        output = torch.nn.functional.relu(output)

        # Apply the second fully connected layer
        output = self.dense2(output)

        return output
