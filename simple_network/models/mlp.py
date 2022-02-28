import torch
from simple_network.nn import ReLu

class MLP:
    def __init__(self, in_channels, hidden_channels, out_channels):
        self.W1 = torch.nn.Parameter(torch.randn(
            in_channels, hidden_channels, requires_grad=True) * 0.01)
        self.b1 = torch.nn.Parameter(torch.zeros(hidden_channels, requires_grad=True))
        self.W2 = torch.nn.Parameter(torch.randn(
            hidden_channels, out_channels, requires_grad=True) * 0.01)
        self.b2 = torch.nn.Parameter(torch.zeros(out_channels, requires_grad=True))
        self.relu = ReLu()

    def __call__(self, x):
        x = torch.matmul(x, self.W1) + self.b1
        x = self.relu(x)
        return torch.matmul(x, self.W2) + self.b2

    def parameters(self):
        return [self.W1, self.b1, self.W2, self.b2]

