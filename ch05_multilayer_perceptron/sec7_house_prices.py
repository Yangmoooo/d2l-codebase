import numpy as np
import polars as pl
import polars.selectors as cs
import torch
from torch import Tensor, nn
from torch.types import Number
from torch.utils.data import DataLoader, TensorDataset


def preprocess_data(train_path: str, test_path: str) -> tuple[Tensor, Tensor, Tensor]:
    """读取并清洗数据"""
    # 读取成 DataFrame
    train_df = pl.read_csv(train_path, null_values=["NA"])
    test_df = pl.read_csv(test_path, null_values=["NA"])

    # 提取标签
    train_labels = train_df.select("SalePrice").to_numpy().reshape(-1, 1)

    # 剔除 ID 和 Label 列
    features_train = train_df.drop(["Id", "SalePrice"])
    features_test = test_df.drop(["Id"])

    # 垂直拼接
    all_features = pl.concat([features_train, features_test])

    # 选择所有数字类型的列，标准化为 (x - mean) / std，然后把 null 填为 0
    numeric_features = all_features.select(cs.numeric()).columns
    all_features = all_features.with_columns(
        [
            ((pl.col(col) - pl.col(col).mean()) / pl.col(col).std()).fill_null(0).alias(col)
            for col in numeric_features
        ]
    )

    # 处理所有字符串/分类类型的列，但 Polars 的 to_dummies 默认不处理 null，需要先把 null 填充为 "NaN"
    str_features = all_features.select(cs.string() | cs.categorical()).columns
    all_features = all_features.with_columns([pl.col(col).fill_null("NaN") for col in str_features])
    # 一键 One-Hot，结果会自动展开
    all_features = all_features.to_dummies(str_features)

    # 转换回 Tensor，切分训练集和测试集
    n_train = train_df.height
    train_features_tensor = torch.tensor(all_features[:n_train].to_numpy(), dtype=torch.float32)
    test_features_tensor = torch.tensor(all_features[n_train:].to_numpy(), dtype=torch.float32)
    train_labels_tensor = torch.log1p(torch.tensor(train_labels, dtype=torch.float32))

    return train_features_tensor, test_features_tensor, train_labels_tensor


def log_rmse(model: nn.Module, features: Tensor, labels: Tensor) -> Number:
    criterion = nn.MSELoss()
    with torch.no_grad():
        # 将预测值限制在 [1,+∞) 之间
        preds = torch.clamp(model(features), 1, float("inf"))
        # 核心思想是认定将 100w 错估为 200w 与 100 错估为 200 是一样严重的错误
        rmse = torch.sqrt(criterion(preds, labels))
    return rmse.item()


def get_model(in_features: int) -> nn.Module:
    net = nn.Sequential(nn.Linear(in_features, 1))
    layer = net[0]
    if isinstance(layer, nn.Linear):
        nn.init.normal_(layer.weight, std=0.01)
        nn.init.zeros_(layer.bias)
    return net


def get_k_fold_data(k, i, X, y):
    """
    K-Fold 交叉验证
    根据总折数 k 和当前折数 i 获取训练集和验证集
    """
    assert k > 1

    fold_size = X.shape[0] // k

    valid_start = i * fold_size
    valid_end = (i + 1) * fold_size

    X_valid, y_valid = X[valid_start:valid_end], y[valid_start:valid_end]

    X_train = torch.cat([X[:valid_start], X[valid_end:]], dim=0)
    y_train = torch.cat([y[:valid_start], y[valid_end:]], dim=0)
    return X_train, y_train, X_valid, y_valid


def train_k_fold(k, X_train, y_train, num_epochs, learning_rate, weight_decay, batch_size, device):
    train_loss_sum, valid_loss_sum = 0, 0
    for i in range(k):
        data = get_k_fold_data(k, i, X_train, y_train)
        train_features_fold, train_labels_fold, valid_features_fold, valid_labels_fold = [
            d.to(device) for d in data
        ]

        model = get_model(X_train.shape[1]).to(device)

        loss_fn = nn.MSELoss()
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )

        train_iter = DataLoader(
            TensorDataset(train_features_fold, train_labels_fold), batch_size, shuffle=True
        )

        for _ in range(num_epochs):
            model.train()
            for X, y in train_iter:
                optimizer.zero_grad()
                loss = loss_fn(model(X), y)
                loss.backward()
                optimizer.step()

        # 训练时可以用简单的 MSE，但评估时要用 kaggle 官方指标 LogRMSE
        train_loss = log_rmse(model, train_features_fold, train_labels_fold)
        valid_loss = log_rmse(model, valid_features_fold, valid_labels_fold)

        train_loss_sum += train_loss
        valid_loss_sum += valid_loss

        print(f"Fold {i + 1}, Train LogRMSE {train_loss:.4f}, Valid LogRMSE {valid_loss:.4f}")

    return train_loss_sum / k, valid_loss_sum / k


def main():
    k = 5
    num_epochs = 100
    lr = 0.01
    weight_decay = 0.1
    batch_size = 64

    train_path = "./data/house-prices-advanced-regression-techniques/train.csv"
    test_path = "./data/house-prices-advanced-regression-techniques/test.csv"
    save_path = "./data/house-prices-advanced-regression-techniques/submission.csv"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on {device}")

    train_features, test_features, train_labels = preprocess_data(train_path, test_path)
    print(f"Features dimension: {train_features.shape[1]}")

    train_l, valid_l = train_k_fold(
        k, train_features, train_labels, num_epochs, lr, weight_decay, batch_size, device
    )
    print(
        f"{k}-fold validation: Avg Train LogRMSE: {train_l:.4f}, Avg Valid LogRMSE: {valid_l:.4f}"
    )

    # 最终预测，通过上面的调整好参数后使用全部训练数据重新训练一次
    print("Retraining on full dataset...")
    model = get_model(train_features.shape[1]).to(device)
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    train_iter = DataLoader(
        TensorDataset(train_features.to(device), train_labels.to(device)), batch_size, shuffle=True
    )

    for _ in range(num_epochs):
        model.train()
        for X, y in train_iter:
            optimizer.zero_grad()
            loss = loss_fn(model(X), y)
            loss.backward()
            optimizer.step()

    # 生成提交文件
    model.eval()
    preds = model(test_features.to(device)).detach().cpu().numpy().reshape(-1)
    preds = np.expm1(preds)

    ids = pl.read_csv(test_path, null_values=["NA"])["Id"]
    submission = pl.DataFrame({"Id": ids, "SalePrice": preds})
    submission.write_csv(save_path)
    print("Submission saved to submission.csv")


if __name__ == "__main__":
    main()
