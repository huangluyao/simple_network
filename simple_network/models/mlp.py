import torch
import torch.nn as nn


class MLP:
    def __init__(self, in_channels, hidden_channels, out_channels):
        self.W1 = nn.Parameter(torch.randn(
            in_channels, hidden_channels, requires_grad=True) * 0.01)
        self.b1 = nn.Parameter(torch.zeros(hidden_channels, requires_grad=True))
        self.W2 = nn.Parameter(torch.randn(
            hidden_channels, out_channels, requires_grad=True) * 0.01)
        self.b2 = nn.Parameter(torch.zeros(out_channels, requires_grad=True))

    def __call__(self, x):

        pass

    def parameters(self):
        return [self.W1, self.b1, self.W2, self.b2]

