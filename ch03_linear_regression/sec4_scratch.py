import random
from collections.abc import Iterator

import torch
from torch import Tensor

from my_utils.data import synthetic_data


def data_iter(batch_size: int, features: Tensor, labels: Tensor) -> Iterator[tuple[Tensor, Tensor]]:
    """手写的小批量数据生成器 as torch.utils.data.DataLoader"""
    num_examples = len(features)
    indices = list(range(num_examples))
    random.shuffle(indices)

    for i in range(0, num_examples, batch_size):
        batch_indices = torch.tensor(indices[i : min(i + batch_size, num_examples)])
        yield features[batch_indices], labels[batch_indices]


def linreg(X: Tensor, w: Tensor, b: Tensor) -> Tensor:
    """线性回归模型 y = Xw + b"""
    return X @ w + b


def squared_loss(y_hat: Tensor, y: Tensor) -> Tensor:
    """均方损失函数 MSE"""
    # reshape 是为了保证维度一致，防止广播机制导致的错误
    return (y_hat - y.reshape(y_hat.shape)) ** 2 / 2


def sgd(params: list[Tensor], lr: float, batch_size: int) -> None:
    """小批量随机梯度下降"""
    # 更新参数时不需要计算梯度，必须用 no_grad 暂时关闭自动求导
    with torch.no_grad():
        for param in params:
            grad = param.grad  # 这里的 param.grad 是 .backward() 自动算出来的
            if grad is not None:
                param -= lr * grad / batch_size  # 根据梯度更新参数
                grad.zero_()  # PyTorch 会累积梯度，作为标准 SGD 每次更新完必须清零


def main():
    # --- 初始化真实参数 ---
    true_w = torch.tensor([2, -3.4])
    true_b = 4.2
    features, labels = synthetic_data(true_w, true_b, 1000)

    # --- 初始化模型参数 ---
    # requires_grad=True 就是通知 PyTorch 关注该变量，它是需要求导的
    w = torch.normal(0, 0.01, size=(2, 1), requires_grad=True)
    b = torch.zeros(1, requires_grad=True)

    # --- 设置超参数 ---
    lr = 0.03
    num_epochs = 3
    batch_size = 10

    print(f"Start training on {len(features)} samples...")

    # --- 训练循环 ---
    for epoch in range(num_epochs):
        for X, y in data_iter(batch_size, features, labels):
            # 前向传播 Forward
            y_hat = linreg(X, w, b)

            # 计算损失 Calculate Loss
            loss = squared_loss(y_hat, y)

            # 反向传播 Backward
            loss.sum().backward()  # sum() 是为了把向量变成标量，backward() 会自动更新 w.grad 和 b.grad

            # 参数更新 Update
            sgd([w, b], lr, batch_size)

        # 每个 epoch 结束后评估一下
        with torch.no_grad():
            loss = squared_loss(linreg(features, w, b), labels)
            print(f"Epoch {epoch + 1}, loss {float(loss.mean()):f}")

    # --- 结果对比 ---
    print(f"\n真实 w: {true_w}")
    print(f"预测 w: {w.reshape(true_w.shape).detach()}")
    print(f"真实 b: {true_b}")
    print(f"预测 b: {b.item()}")


if __name__ == "__main__":
    main()
