import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from my_utils.data import load_data_fashion_mnist


def init_weights(m: nn.Module):
    """权重初始化函数"""
    if isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, std=0.01)
        nn.init.zeros_(m.bias)


def accuracy(y_hat: torch.Tensor, y: torch.Tensor) -> float:
    """计算预测正确的数量"""
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = y_hat.argmax(dim=1)

    cmp = y_hat.type(y.dtype) == y
    return float(cmp.type(y.dtype).sum())


def evaluate_accuracy(net: nn.Module, data_iter: DataLoader) -> float:
    """计算模型在指定数据集上的精度"""
    if isinstance(net, nn.Module):
        net.eval()  # 切换到评估模式，会影响 Dropout/BatchNorm 等层

    metric = [0.0, 0.0]
    with torch.no_grad():
        for X, y in data_iter:
            y_hat = net(X)
            metric[0] += accuracy(y_hat, y)
            metric[1] += y.numel()
    return metric[0] / metric[1]


def main():
    batch_size = 256
    learning_rate = 0.1
    num_epochs = 10

    train_iter, test_iter = load_data_fashion_mnist(batch_size)

    # nn.Flatten() 会保留第 0 维 batch，拉直后面所有维
    # 即自动把 (N, 1, 28, 28) 展平成 (N, 784)
    net = nn.Sequential(nn.Flatten(), nn.Linear(784, 10))
    net.apply(init_weights)  # 遍历每一层进行初始化

    # nn.CrossEntropyLoss 的内部 = LogSoftmax + NLLLoss
    # 所以 net 的输出不需要单独的 softmax，直接输出 Logits 即可
    # 默认 reduction='mean'，会自动除以 batch_size
    loss_fn = nn.CrossEntropyLoss()

    optimizer = optim.SGD(net.parameters(), lr=learning_rate)

    print("Start training Softmax Regression (Concise)...")

    for epoch in range(num_epochs):
        # 切换到训练模式
        net.train()

        # [累加 Loss, 累加正确数, 样本数]
        metric = [0.0, 0.0, 0.0]

        for X, y in train_iter:
            # 前向传播
            y_hat = net(X)

            # 计算损失
            loss = loss_fn(y_hat, y)

            # 梯度清零
            optimizer.zero_grad()

            # 反向传播
            loss.backward()

            # 更新参数
            optimizer.step()

            # 记录数据，这里的 loss 是 mean 过的，所以要 * len(y) 还原回 sum
            with torch.no_grad():
                metric[0] += float(loss) * y.shape[0]
                metric[1] += accuracy(y_hat, y)
                metric[2] += y.shape[0]

        train_loss = metric[0] / metric[2]
        train_acc = metric[1] / metric[2]
        test_acc = evaluate_accuracy(net, test_iter)

        print(
            f"Epoch {epoch + 1}: Loss {train_loss:.3f}, Train Acc {train_acc:.3f}, Test Acc {test_acc:.3f}"
        )


if __name__ == "__main__":
    main()
