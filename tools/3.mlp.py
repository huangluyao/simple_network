import os
import torch
import simple_network
import simple_network.nn.functional as F
from simple_network.dataset import load_data_fashion_mnist, get_fashion_mnist_labels
from simple_network.utils import RealTimeGraph, show_images, Accumulator


def accuracy(y_hat, y):
    """计算预测正确的数量"""
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = y_hat.argmax(axis=1)
    cmp = y_hat.type(y.dtype) == y
    return float(cmp.type(y.dtype).sum())


def train_one_epoch(net, train_iter, optimizer):
    """网络训练一个epoch的操作"""
    metric = Accumulator(3)  # 累加 损失值， 准确度， 总数 三个变量

    for X, y in train_iter:
        b, _, _, _ = X.shape
        X = X.reshape(b, -1)    # 对二维图像展平成1维数据
        y_hat = net(X)
        loss = F.cross_entropy(y_hat, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        metric.add(float(loss.sum()), accuracy(y_hat, y), y.numel())

    return metric[0] / metric[2], metric[1] / metric[2]


def evaluate(net, data_iter):
    metric = Accumulator(2)  # 正确预测数、预测总数
    with torch.no_grad():
        for X, y in data_iter:
            b, _, _, _ = X.shape
            X = X.reshape(b, -1)  # 对二维图像展平成1维数据
            metric.add(accuracy(net(X), y), y.numel())
    return metric[0] / metric[1]


if __name__ == "__main__":

    # 设置相关训练参数
    batch_size = 256
    num_inputs = 784
    hidden_channels = 256
    num_outputs = 10
    lr = 0.1
    num_epochs = 10

    # 创建工作途径
    work_dir = os.path.join("..", "work_dir", "mlp")
    if not os.path.exists(work_dir):
        os.makedirs(work_dir)
    # 加载训练数据和测试数据
    train_iter, test_iter = load_data_fashion_mnist(batch_size)

    net = simple_network.models.MLP(num_inputs, hidden_channels, num_outputs)    # 定义网络
    optimizer = simple_network.optim.SGD(net.parameters(), lr=lr)


    epochs = []
    train_loss_list = []
    train_acc_list = []
    val_acc_list =[]

    acc_path = os.path.join(work_dir, "acc.png")
    rg_acc = RealTimeGraph("train", x_label="epoch", y_label="acc", legend=["train_acc", "test_acc"])
    rg_loss = RealTimeGraph("loss", x_label="epoch", y_label="loss", legend=["loss"])
    # 开始训练
    for epoch in range(num_epochs):

        train_metrics = train_one_epoch(net, train_iter, optimizer)

        test_acc = evaluate(net, test_iter)

        print("epoch {}, loss {}, trian accuracy {} test_acc {}".
              format(epoch, train_metrics[0], train_metrics[1],
                     test_acc))
        epochs.append(epoch)
        train_loss_list.append(train_metrics[0])
        train_acc_list.append(train_metrics[1])
        val_acc_list.append(test_acc)

        rg_acc.add(epochs,  [train_acc_list, val_acc_list])
        rg_loss.add(epochs, [train_loss_list])

    rg_acc.save_fig(os.path.join(work_dir, "acc.png"))
    rg_loss.save_fig(os.path.join(work_dir, "loss.png"))

    for X, y in test_iter:
        b, _, _, _ = X.shape
        X = X.reshape(b, -1)    # 对二维图像展平成1维数据
        break
    trues = get_fashion_mnist_labels(y)
    preds = get_fashion_mnist_labels(net(X).argmax(axis=1))
    titles = [true +'\n' + pred for true, pred in zip(trues, preds)]
    n = 8
    show_images(
        X[0:n].reshape((n, 28, 28)), 2, n//2, titles=titles[0:n],scale=3,
        save_path=os.path.join(work_dir, "result.png"))
