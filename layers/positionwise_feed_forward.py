import torch

class PositionwiseFeedForward(torch.nn.Module):
    def __init__(self, d_model, d_ff):
        super(PositionwiseFeedForward, self).__init__()
        self.dense1 = torch.nn.Linear(d_model, d_ff)
        self.dense2 = torch.nn.Linear(d_ff, d_model)

    def forward(self, x):
        # Apply the first fully connected layer and ReLU activation
        output = self.dense1(x)
        output = torch.nn.functional.relu(output)

        # Apply the second fully connected layer
        output = self.dense2(output)

        return output