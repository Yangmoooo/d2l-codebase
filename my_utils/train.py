from collections.abc import Callable
from typing import Any

import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from my_utils.metric import accuracy


def train_epoch(
    model: nn.Module | Callable[[Tensor], Tensor],
    train_iter: DataLoader,
    loss_fn: nn.Module | Callable[[Tensor, Tensor], Tensor],
    optimizer: torch.optim.Optimizer | Any,
    device: torch.device | str = "cpu",
) -> tuple[float, float]:
    """
    通用的训练一个 epoch 的函数

    Args:
        model: 模型
            - concise: nn.Module 实例
            - scratch: 实现了 __call__ 的类，接受 X 返回 y_hat
        train_iter: 数据迭代器
        loss_fn: 损失函数
            - concise: 如 nn.CrossEntropyLoss 返回标量 mean
            - scratch: 如自定义的 cross_entropy 返回向量 vector
        optimizer: 优化器
            - concise: torch.optim.Optimizer 实例
            - scratch: 实现了 step() 和 zero_grad() 的适配器对象
        device: cpu or cuda

    Returns:
        tuple[float,float]: 平均训练损失和训练准确率
    """
    if isinstance(model, nn.Module):
        model.train()
        model.to(device)

    metric = [0.0, 0.0, 0.0]  # loss, acc, total

    pbar = tqdm(train_iter, desc="Training", leave=False)

    for X, y in pbar:
        X, y = X.to(device), y.to(device)

        # 前向传播
        y_hat = model(X)

        # 计算损失
        loss = loss_fn(y_hat, y)

        # 梯度清零
        optimizer.zero_grad()

        # 反向传播，5.2 及后续代码已将 loss_fn 改为 PyTorch 风格
        loss.backward()

        # 更新参数
        optimizer.step()

        # 记录指标
        with torch.no_grad():
            metric[0] += float(loss) * y.numel()
            metric[1] += accuracy(y_hat, y)
            metric[2] += y.numel()

            current_avg_loss = metric[0] / metric[2]
            current_avg_acc = metric[1] / metric[2]

            pbar.set_postfix({"loss": f"{current_avg_loss:.4f}", "acc": f"{current_avg_acc:.4f}"})

    return metric[0] / metric[2], metric[1] / metric[2]
