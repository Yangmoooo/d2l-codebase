import torch
from torch import nn, optim

from my_utils.data import load_data_fashion_mnist
from my_utils.metric import evaluate_accuracy
from my_utils.train import train_epoch


def get_lenet():
    """
    LeNet-5
    Conv -> Sigmoid -> AvgPool -> Conv -> Sigmoid -> AvgPool -> Flatten -> Linear...
    """
    net = nn.Sequential(
        # --- 第1层 ---
        # 输入: (B, 1, 28, 28)
        # 卷积: kernel=5, padding=2 -> (28 - 5 + 2*2)/1 + 1 = 28
        # 输出: (B, 6, 28, 28)  <-- 6个输出通道，提取6种基础特征
        nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, padding=2),
        nn.Sigmoid(),

        # 池化: kernel=2, stride=2 -> 28 / 2 = 14
        # 输出: (B, 6, 14, 14)
        nn.AvgPool2d(kernel_size=2, stride=2),

        # --- 第2层 ---
        # 卷积: kernel=5
        # 输入: (B, 6, 14, 14) -> (14 - 5)/1 + 1 = 10
        # 输出: (B, 16, 10, 10) <-- 通道数增加到16，组合特征
        nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5),
        nn.Sigmoid(),

        # 池化: 10 / 2 = 5
        # 输出: (B, 16, 5, 5)
        nn.AvgPool2d(kernel_size=2, stride=2),

        # --- 全连接层 ---
        nn.Flatten(), # 拉直: 16 * 5 * 5 = 400

        nn.Linear(400, 120),
        nn.Sigmoid(),

        nn.Linear(120, 84),
        nn.Sigmoid(),

        nn.Linear(84, 10) # 输出 10 类
    )
    return net

def init_weights(m):
    if type(m) is nn.Linear or type(m) is nn.Conv2d:
        nn.init.xavier_uniform_(m.weight)

def main():
    batch_size = 256
    lr = 0.9  # Sigmoid 需要较大的学习率，如果是 ReLU 可以用 0.1
    num_epochs = 10

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on {device}")

    train_iter, test_iter = load_data_fashion_mnist(batch_size=batch_size)

    net = get_lenet().to(device)
    net.apply(init_weights)

    X_dummy = torch.rand(size=(1, 1, 28, 28), device=device)
    print(f"Model output shape check: {net(X_dummy).shape}")

    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=lr)

    print(f"{'Epoch':^10} | {'Train Loss':^12} | {'Train Acc':^12} | {'Test Acc':^12}")
    print("-" * 55)

    for epoch in range(num_epochs):
        train_loss, train_acc = train_epoch(net, train_iter, loss_fn, optimizer, device)
        test_acc = evaluate_accuracy(net, test_iter, device)
        print(f"{epoch+1:^10} | {train_loss:^12.4f} | {train_acc:^12.4f} | {test_acc:^12.4f}")

if __name__ == "__main__":
    main()
