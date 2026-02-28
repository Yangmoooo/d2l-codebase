import torch
from torch import nn, optim

from my_utils.data import load_data_fashion_mnist
from my_utils.metric import evaluate_accuracy
from my_utils.train import train_epoch


def init_weights(m):
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        nn.init.xavier_uniform_(m.weight)


def get_alexnet():
    """
    AlexNet
    Conv(96,11,s4,p1) -> ReLU -> MaxPool(3,s2)
    Conv(256,5,p2)    -> ReLU -> MaxPool(3,s2)
    Conv(384,3,p1)    -> ReLU
    Conv(384,3,p1)    -> ReLU
    Conv(256,3,p1)    -> ReLU -> MaxPool(3,s2)
    Flatten -> Linear(4096) -> ReLU -> Dropout
            -> Linear(4096) -> ReLU -> Dropout
            -> Linear(10)
    """
    net = nn.Sequential(
        nn.Conv2d(1, 96, kernel_size=11, stride=4, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=3, stride=2),
        nn.Conv2d(96, 256, kernel_size=5, padding=2),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=3, stride=2),
        nn.Conv2d(256, 384, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.Conv2d(384, 384, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.Conv2d(384, 256, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=3, stride=2),
        nn.Flatten(),
        nn.Linear(6400, 4096),  # 256 * 5 * 5 = 6400
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
    batch_size = 256
    lr = 0.001
    num_epochs = 10

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on {device}")

    train_iter, test_iter = load_data_fashion_mnist(batch_size=batch_size, resize=224)

    net = get_alexnet().to(device)

    X_dummy = torch.rand(size=(1, 1, 224, 224), device=device)
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
