import torch
from torch import Tensor

from my_utils.activations import relu, softmax
from my_utils.data import load_data_fashion_mnist
from my_utils.losses import cross_entropy
from my_utils.metric import evaluate_accuracy
from my_utils.optim import SGD
from my_utils.train import train_epoch


class MLP:
    def __init__(self, W1: Tensor, b1: Tensor, W2: Tensor, b2: Tensor):
        self.W1 = W1
        self.b1 = b1
        self.W2 = W2
        self.b2 = b2

    def __call__(self, X: Tensor):
        flatten = X.reshape((-1, 784))
        hidden = relu(flatten @ self.W1 + self.b1)
        output = hidden @ self.W2 + self.b2
        return softmax(output)


def main():
    batch_size = 256
    lr = 0.1
    num_epochs = 10

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on {device}")

    train_iter, test_iter = load_data_fashion_mnist(batch_size)

    num_inputs, num_hiddens, num_outputs = 784, 256, 10

    W1 = torch.normal(0, 0.01, size=(num_inputs, num_hiddens), requires_grad=True)
    b1 = torch.zeros(num_hiddens, requires_grad=True)
    W2 = torch.normal(0, 0.01, size=(num_hiddens, num_outputs), requires_grad=True)
    b2 = torch.zeros(num_outputs, requires_grad=True)
    params = [W1, b1, W2, b2]

    model = MLP(*params)
    optimizer = SGD(params, lr)

    print(f"{'Epoch':^10} | {'Train Loss':^12} | {'Train Acc':^12} | {'Test Acc':^12}")
    print("-" * 55)

    for epoch in range(num_epochs):
        train_loss, train_acc = train_epoch(model, train_iter, cross_entropy, optimizer, device)
        test_acc = evaluate_accuracy(model, test_iter, device)
        print(f"{epoch + 1:^10} | {train_loss:^12.4f} | {train_acc:^12.4f} | {test_acc:^12.4f}")


if __name__ == "__main__":
    main()
