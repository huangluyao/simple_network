import torch


class LinearReg:

    def __init__(self, in_channels, out_channels):
        self.w = torch.normal(0, 0.01, size=(in_channels, out_channels), requires_grad=True)
        self.b = torch.zeros(out_channels, requires_grad=True)

    def __call__(self, X):
        return torch.matmul(X, self.w) + self.b

    def parameters(self):
        return [self.w, self.b]

