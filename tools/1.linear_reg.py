import torch
import random
import matplotlib.pyplot as plt
import simple_network.nn as nn
from simple_network.models import LinearReg
from simple_network.optim import SGD


def sythetic_data(w, b, num_examples):
    #  生成 y = wX + b + 噪声
    X = torch.normal(0, 1, (num_examples, len(w)))
    y = torch.matmul(X, w) + b
    y += torch.normal(0, 0.01, y.shape)
    return X, y.reshape((-1, 1))

def data_iter(batch_size, features, labels):

    num_examples = len(features)
    indices = list(range(num_examples))
    random.shuffle(indices)

    for i in range(0, num_examples, batch_size):

        batch_indices = torch.tensor(indices[i:min(i+batch_size, num_examples)])
        yield features[batch_indices], labels[batch_indices]

if __name__ == "__main__":

    # 1. 设置真实值 w 和 b
    true_w = torch.tensor([2, -3.4])
    ture_b = 4.2
    # 2. 对数据添加噪声
    features, labels = sythetic_data(true_w, ture_b, 1000)
    # 3. 可视化数据
    plt.figure()
    plt.scatter(features[:, 1].detach().numpy(),
                labels.detach().numpy(), 1
                )
    plt.show()

    # 4. 开始训练线性回归
    batch_size = 10
    lr = 0.03
    num_epochs = 3
    net = LinearReg(2, 1) # 创建网络模型
    loss = nn.MSELoss() # 创建损失函数
    optim = SGD(net.parameters(), lr=lr) # 创建优化器

    for epoch in range(num_epochs):
        for X, y in data_iter(batch_size, features, labels):
            y_hat = net(X)
            l = loss(y_hat, y)
            optim.zero_grad()
            l.backward()
            optim.step()

        with torch.no_grad():
            train_l = loss(net(features), labels)
            print("epoch {}, loss {}".format(epoch + 1, float(train_l)))

    # 打印网络推理出来的数值
    print("network w ={}".format(net.w.detach().numpy()))
    print("network b ={}".format(net.b.detach().numpy()))
