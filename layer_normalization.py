import numpy as np

class LayerNormalization:
    def __init__(self, d_model, epsilon=1e-5):
        self.epsilon = epsilon
        self.scale = np.ones(d_model)
        self.bias = np.zeros(d_model)

    def forward(self, x):
        mean = np.mean(x, axis=-1, keepdims=True)
        variance = np.var(x, axis=-1, keepdims=True)
        normalized = (x - mean) / np.sqrt(variance + self.epsilon)
        output = self.scale * normalized + self.bias
        return output
