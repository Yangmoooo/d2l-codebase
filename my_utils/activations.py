import torch
from torch import Tensor


def relu(X: Tensor) -> Tensor:
    a = torch.zeros_like(X)
    return torch.max(X, a)


def sigmoid(X: Tensor) -> Tensor:
    return 1 / (1 + torch.exp(-X))


def softmax(X: Tensor) -> Tensor:
    X_max = X.max(dim=1, keepdim=True).values
    X_exp = torch.exp(X - X_max)
    partition = X_exp.sum(dim=1, keepdim=True)
    return X_exp / partition


def dropout(X: Tensor, probability: float) -> Tensor:
    assert 0 <= probability <= 1
    if probability == 0:
        return X
    mask = (torch.rand(X.shape, device=X.device) > probability).float()
    return mask * X / (1.0 - probability)
