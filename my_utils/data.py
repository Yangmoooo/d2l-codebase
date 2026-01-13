import torch
import torchvision
from torch import Tensor
from torch.utils.data import DataLoader
from torchvision import transforms


def synthetic_data(w: Tensor, b: float | Tensor, num_examples: int) -> tuple[Tensor, Tensor]:
    """生成 y = Xw + b + 噪声"""
    X = torch.normal(0, 1, (num_examples, len(w)))
    y = X @ w + b
    y += torch.normal(0, 0.01, y.shape)
    return X, y.reshape((-1, 1))


def load_data_fashion_mnist(
    batch_size: int,
    resize: int | None = None,
    root: str = "./data",
) -> tuple[DataLoader, DataLoader]:
    """
    下载并加载 Fashion-MNIST 数据集

    Args:
        batch_size: 批量大小
        resize: 将图片调整为 resize x resize 大小
        root: 数据存储路径
    """
    trans_list: list = [transforms.ToTensor()]
    if resize:
        trans_list.insert(0, transforms.Resize(resize))

    trans = transforms.Compose(trans_list)

    mnist_train = torchvision.datasets.FashionMNIST(
        root=root, train=True, transform=trans, download=True
    )
    mnist_test = torchvision.datasets.FashionMNIST(
        root=root, train=False, transform=trans, download=True
    )

    return (
        DataLoader(mnist_train, batch_size, shuffle=True, num_workers=4),
        DataLoader(mnist_test, batch_size, shuffle=False, num_workers=4),
    )
