import torch


def softmax(X, dim=1):
    X_exp = torch.exp(X)
    partition = X_exp.sum(dim=dim, keepdim=True)
    return X_exp / partition

def cross_entropy(y_hat, y, reduction="mean"):
    y_pred = softmax(y_hat, dim=-1)
    loss = -torch.log(y_pred[range(len(y)), y])

    if reduction == "mean":
        return loss.mean()
    elif reduction == "sum":
        return loss.sum()
    else:
        return loss


if __name__ == "__main__":
    import torch.nn.functional as F
    X = torch.normal(0, 1, (2, 5))
    print(softmax(X, dim=1))
    print(F.softmax(X, dim=1))

    y = torch.tensor([0, 2])
    y_hat = torch.tensor([[0.1, 0.3, 0.6], [0.3, 0.2, 0.5]])


    print(cross_entropy(y_hat, y))
    print(F.cross_entropy(y_hat, y))  # softmax + log + nll loss
