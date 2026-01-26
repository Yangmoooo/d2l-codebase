import torch
from torch import nn

from my_utils.data import load_data_fashion_mnist
from my_utils.metric import evaluate_accuracy
from my_utils.train import train_epoch


def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, std=0.01)
        nn.init.zeros_(m.bias)


def main():
    batch_size = 256
    lr = 0.1
    num_epochs = 10
    dropout_rate = 0.5

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on {device}")

    train_iter, test_iter = load_data_fashion_mnist(batch_size)

    model = nn.Sequential(
        nn.Flatten(),
        nn.Linear(784, 256),
        nn.ReLU(),
        nn.Dropout(p=dropout_rate),
        nn.Linear(256, 10),
    )
    model.apply(init_weights)

    loss = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    print(f"{'Epoch':^10} | {'Train Loss':^12} | {'Train Acc':^12} | {'Test Acc':^12}")
    print("-" * 55)

    for epoch in range(num_epochs):
        train_loss, train_acc = train_epoch(model, train_iter, loss, optimizer, device)
        test_acc = evaluate_accuracy(model, test_iter, device)
        print(f"{epoch + 1:^10} | {train_loss:^12.4f} | {train_acc:^12.4f} | {test_acc:^12.4f}")


if __name__ == "__main__":
    main()
