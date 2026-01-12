import torch
from torch import Tensor


def synthetic_data(w: Tensor, b: float | Tensor, num_examples: int) -> tuple[Tensor, Tensor]:
    """生成 y = Xw + b + 噪声"""
    X = torch.normal(0, 1, (num_examples, len(w)))
    y = X @ w + b
    y += torch.normal(0, 0.01, y.shape)
    return X, y.reshape((-1, 1))
