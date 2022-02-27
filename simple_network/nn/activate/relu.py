import torch

class ReLu:
    def __call__(self, x):
        return  torch.clamp(x, min=0)