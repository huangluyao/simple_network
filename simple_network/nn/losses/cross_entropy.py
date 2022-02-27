import torch
from ..funcational import cross_entropy


class CrossEntropyLoss:

    def __init__(self, reduction="mean"):
        self.reduction = reduction

    def __call__(self, y_hat, y):
        return cross_entropy(y_hat, y, self.reduction)


