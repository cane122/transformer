import numpy as np

class PositionwiseFeedForward:
    def __init__(self, d_model, d_ff):
        self.dense1 = np.random.randn(d_model, d_ff)  # The first fully connected layer
        self.dense2 = np.random.randn(d_ff, d_model)  # The second fully connected layer

    def forward(self, x):
        # Apply the first fully connected layer and ReLU activation
        output = np.dot(x, self.dense1)
        output = np.maximum(output, 0)  # ReLU activation

        # Apply the second fully connected layer
        output = np.dot(output, self.dense2)

        return output
