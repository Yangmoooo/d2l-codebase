from collections.abc import Callable

import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader


def accuracy(y_hat: Tensor, y: Tensor) -> float:
    """计算预测正确的数量"""
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = y_hat.argmax(dim=1)
    cmp = y_hat.type(y.dtype) == y
    return float(cmp.type(y.dtype).sum())


def evaluate_accuracy(
    net: nn.Module | Callable[[Tensor], Tensor],
    data_iter: DataLoader,
    device: torch.device | str = "cpu",
) -> float:
    """计算在指定数据集上的精度"""
    # 如果是 nn.Module 需要设置为评估模式
    if isinstance(net, nn.Module):
        net.eval()
        net.to(device)
    metric = [0.0, 0.0]
    with torch.no_grad():
        for X, y in data_iter:
            X, y = X.to(device), y.to(device)
            y_hat = net(X)
            metric[0] += accuracy(y_hat, y)
            metric[1] += y.numel()
    return metric[0] / metric[1]
