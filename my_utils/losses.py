import torch
from torch import Tensor


def squared_loss(y_hat: Tensor, y: Tensor) -> Tensor:
    """均方误差"""
    return (y_hat - y.reshape(y_hat.shape)) ** 2 / 2


def cross_entropy(y_hat: Tensor, y: Tensor) -> Tensor:
    """交叉熵损失"""
    # 使用高级索引提取正确类别的概率
    # + 1e-10 是为了防止 log(0)
    # .mean() 是为了与 PyTorch 的 nn.CrossEntropyLoss 默认行为对齐
    return -torch.log(y_hat[range(len(y_hat)), y] + 1e-10).mean()
