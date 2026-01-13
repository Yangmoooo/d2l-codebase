from collections.abc import Callable

import torch
from torch import Tensor
from torch.utils.data import DataLoader

from my_utils.data import load_data_fashion_mnist


def softmax(X: Tensor) -> Tensor:
    """softmax 函数"""
    X_max = X.max(dim=1, keepdim=True).values
    X_exp = torch.exp(X - X_max)
    partition = X_exp.sum(dim=1, keepdim=True)
    return X_exp / partition  # 广播机制


def net(X: Tensor, W: Tensor, b: Tensor) -> Tensor:
    """softmax 模型 forward"""
    # -1 表示 batch_size, W.shape[0] 是 784
    X = X.reshape((-1, W.shape[0]))  # (batch_size, 1, 28, 28) -> (batch_size, 784)
    return softmax(X @ W + b)


def cross_entropy(y_hat: Tensor, y: Tensor) -> Tensor:
    """交叉熵损失"""
    # 整数数组索引，给定一组行坐标和一组列坐标，选出若干具体的点
    # range(len(y_hat)): 生成 [0, 1, ..., batch-1] 作为行索引
    # y: 真实标签作为列索引
    # y_hat[range, y] 就会得到所有样本的真实类别对应的预测概率
    return -torch.log(y_hat[range(len(y_hat)), y])


def accuracy(y_hat: Tensor, y: Tensor) -> float:
    """计算预测正确的数量"""
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = y_hat.argmax(dim=1)  # dim=1 取每一行（即每个样本）最大值的索引

    cmp = y_hat.type(y.dtype) == y
    return float(cmp.type(y.dtype).sum())


def evaluate_accuracy(
    net: Callable[[Tensor, Tensor, Tensor], Tensor], data_iter: DataLoader, W: Tensor, b: Tensor
) -> float:
    """计算在指定数据集上的精度"""
    if isinstance(net, torch.nn.Module):
        net.eval()  # 如果是 nn.Module 需要设为评估模式

    metric = [0.0, 0.0]  # [累加正确数, 累加总样本数]
    with torch.no_grad():
        for X, y in data_iter:
            y_hat = net(X, W, b)
            metric[0] += accuracy(y_hat, y)
            metric[1] += y.numel()
    return metric[0] / metric[1]


def sgd(params: list[Tensor], lr: float, batch_size: int) -> None:
    """小批量随机梯度下降"""
    with torch.no_grad():
        for param in params:
            grad = param.grad
            if grad is not None:
                param -= lr * grad / batch_size
                grad.zero_()


def main():
    batch_size = 256
    learning_rate = 0.1
    num_epochs = 10

    train_iter, test_iter = load_data_fashion_mnist(batch_size)

    num_inputs = 784
    num_outputs = 10

    W = torch.normal(0, 0.01, size=(num_inputs, num_outputs), requires_grad=True)
    b = torch.zeros(num_outputs, requires_grad=True)

    print("Start training on Fashion-MNIST (Scratch)...")

    for epoch in range(num_epochs):
        # [累加 Loss, 累加正确数, 样本总数]
        metric = [0.0, 0.0, 0.0]

        for X, y in train_iter:
            # Forward
            y_hat = net(X, W, b)

            # Loss
            loss = cross_entropy(y_hat, y)

            # Backward
            loss.sum().backward()

            # Optimize
            sgd([W, b], learning_rate, batch_size)

            # Metric
            with torch.no_grad():
                metric[0] += float(loss.sum())
                metric[1] += accuracy(y_hat, y)
                metric[2] += y.numel()

        train_loss = metric[0] / metric[2]
        train_acc = metric[1] / metric[2]
        test_acc = evaluate_accuracy(net, test_iter, W, b)

        print(
            f"Epoch {epoch + 1}: Loss {train_loss:.3f}, Train Acc {train_acc:.3f}, Test Acc {test_acc:.3f}"
        )


if __name__ == "__main__":
    main()
