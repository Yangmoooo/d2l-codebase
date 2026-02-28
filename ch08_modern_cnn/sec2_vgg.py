import torch
from torch import nn, optim

from my_utils.data import load_data_fashion_mnist
from my_utils.metric import evaluate_accuracy
from my_utils.train import train_epoch


def init_weights(m):
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        nn.init.xavier_uniform_(m.weight)


def vgg_block(num_convs: int, in_channels: int, out_channels: int) -> nn.Sequential:
    """
    VGG Block = n * Conv(3x3,pad=1) + ReLU + MaxPool(2,2)
    每个块内所有卷积的输出通道数相同，只有第一个卷积需要做通道变换
    """
    layers = []
    for i in range(num_convs):
        layers.append(
            nn.Conv2d(
                in_channels if i == 0 else out_channels, out_channels, kernel_size=3, padding=1
            )
        )
        layers.append(nn.ReLU())
    layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
    return nn.Sequential(*layers)


def get_vgg():
    """
    VGG-11
    输入: (B, 1, 96, 96)

    VGG块配置: (num_convs, out_channels)
    Block1: 1 Conv(64)  -> MaxPool   96 ->  48
    Block2: 1 Conv(128) -> MaxPool   48 ->  24
    Block3: 2 Conv(256) -> MaxPool   24 ->  12
    Block4: 2 Conv(512) -> MaxPool   12 ->   6
    Block5: 2 Conv(512) -> MaxPool    6 ->   3

    Flatten: 512 * 3 * 3 = 4608
    Linear(4096) -> ReLU -> Dropout
    Linear(4096) -> ReLU -> Dropout
    Linear(10)
    """
    # (num_convs, out_channels)
    arch = [(1, 64), (1, 128), (2, 256), (2, 512), (2, 512)]

    conv_blocks = []
    in_channels = 1  # Fashion-MNIST 单通道
    for num_convs, out_channels in arch:
        conv_blocks.append(vgg_block(num_convs, in_channels, out_channels))
        in_channels = out_channels

    net = nn.Sequential(
        *conv_blocks,
        nn.Flatten(),
        nn.Linear(512 * 3 * 3, 4096),
        nn.ReLU(),
        nn.Dropout(p=0.5),
        nn.Linear(4096, 4096),
        nn.ReLU(),
        nn.Dropout(p=0.5),
        nn.Linear(4096, 10),
    )
    net.apply(init_weights)
    return net


def main():
    batch_size = 64  # VGG 参数较多，使用较小的 batch size
    lr = 0.001
    num_epochs = 10

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on {device}")

    train_iter, test_iter = load_data_fashion_mnist(batch_size=batch_size, resize=96)

    net = get_vgg().to(device)

    X_dummy = torch.rand(size=(1, 1, 96, 96), device=device)
    print(f"Model output shape check: {net(X_dummy).shape}")

    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=lr)

    print(f"{'Epoch':^10} | {'Train Loss':^12} | {'Train Acc':^12} | {'Test Acc':^12}")
    print("-" * 55)

    for epoch in range(num_epochs):
        train_loss, train_acc = train_epoch(net, train_iter, loss_fn, optimizer, device)
        test_acc = evaluate_accuracy(net, test_iter, device)
        print(f"{epoch + 1:^10} | {train_loss:^12.4f} | {train_acc:^12.4f} | {test_acc:^12.4f}")


if __name__ == "__main__":
    main()
