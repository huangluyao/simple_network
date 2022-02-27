import torch

class SGD:

    def __init__(self, params, lr):
        self.params = params
        self.lr = lr

    def zero_grad(self):
        for param in self.params:
            if param.grad is not None:
                param.grad.zero_()

    def step(self):
        with torch.no_grad():
            for param in self.params:
                param -= self.lr * param.grad
