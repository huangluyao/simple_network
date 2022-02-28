from ..functional import softmax

class Softmax:
    def __init__(self, dim=1):
        self.dim = dim

    def __call__(self, X):
        return softmax(X, self.dim)
