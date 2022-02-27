

class MSELoss:
    """
    均方差损失函数
    """
    def __init__(self):
        pass

    def __call__(self, y_hat, y):
        return squared_loss(y_hat, y)

def squared_loss(y_hat, y):
    return ((y_hat - y.reshape(y_hat.shape)) ** 2).mean() / 2

