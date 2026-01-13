import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset

from my_utils.data import synthetic_data


def main():
    true_w = torch.tensor([2, -3.4])
    true_b = 4.2
    features, labels = synthetic_data(true_w, true_b, 1000)

    # TensorDataset 把 X 和 y 绑在一起
    # 使用 DataLoader 替代手写生成器，自动处理 Shuffle Batching 及多进程加载
    dataset = TensorDataset(features, labels)
    data_iter = DataLoader(dataset, batch_size=10, shuffle=True)

    # 使用 nn.Linear 替代手写 linreg
    layer1 = nn.Linear(in_features=2, out_features=1)
    nn.init.normal_(layer1.weight, mean=0, std=0.01)  # 正态分布初始化权重
    nn.init.constant_(layer1.bias, val=0)  # 偏差初始化为 0

    # nn.Sequential 是一个容器，虽然这里只有一层，但习惯上这样写方便扩展
    net = nn.Sequential(layer1)

    # 使用 nn.MSELoss 替代手写 squared_loss
    # 默认 reduction='mean'，即自动求均值，所以下面 SGD 不用除以 batch_size
    loss_fn = nn.MSELoss()

    # 使用 optim.SGD 替代手写 sgd
    # net.parameters() 自动把模型里所有的 w 和 b 传给优化器
    optimizer = optim.SGD(net.parameters(), lr=0.03)

    print("Start training using PyTorch High-level APIs...")

    num_epochs = 3
    for epoch in range(num_epochs):
        for X, y in data_iter:
            # 前向传播
            y_hat = net(X)

            # 计算损失
            loss = loss_fn(y_hat, y)

            # 梯度清零
            # 一般来说都会先清零梯度，但在 scratch 实现中是把清零这一步封装到了参数更新里
            # 二者在效果上是等价的，不过显式调用有更强的操控力
            # 而且旧梯度也有参考价值，比如优化器可能使用动量，虽然 SGD 是没有的
            # 或者在显存不够的场景，可以通过梯度累计攒多个 batch 再更新一次参数
            optimizer.zero_grad()

            # 反向传播
            # 链式法则是执果索因，所以必须从最末端开始，也就是最终计算出的 loss
            # 然后顺着链路反向传导回每一个参数节点，计算出对应的梯度
            loss.backward()

            # 更新参数
            # 优化器是参数的管理者，它知道具体的更新策略，应该如何修改参数
            # 所以 step 和 zero_grad 都是由它调用的
            optimizer.step()

        # 评估
        with torch.no_grad():
            loss = loss_fn(net(features), labels)
            print(f"Epoch {epoch + 1}, loss {loss:f}")

    # --- 结果对比 ---
    params = list(net.parameters())
    w = params[0].data
    b = params[1].data
    print(f"\n真实 w: {true_w}")
    print(f"预测 w: {w.reshape(true_w.shape)}")
    print(f"真实 b: {true_b}")
    print(f"预测 b: {b.item()}")


if __name__ == "__main__":
    main()
