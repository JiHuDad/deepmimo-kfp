"""
train_classifier 컴포넌트 (범용)

features.npy / labels.npy 인터페이스를 사용하는 MLP 분류 모델 학습.
도메인에 무관하게 임의의 분류 문제에 사용 가능하다.

입력 데이터셋 형식:
    train_dataset/features.npy  — (N, D) float32
    train_dataset/labels.npy    — (N,)   int64
    val_dataset/features.npy    — (M, D) float32
    val_dataset/labels.npy      — (M,)   int64

출력:
    output_model/best_model.pt      — 최고 검증 정확도 모델 가중치
    output_model/model_meta.json    — 모델 메타데이터 (input_dim, num_classes, hidden_dims 등)
    output_model/history.json       — 에폭별 loss/accuracy 기록
"""

import os

from kfp import dsl
from kfp.dsl import Input, Output, Dataset, Model, Metrics

_IMAGE_TAG = os.environ.get("IMAGE_TAG", "latest")


@dsl.component(
    base_image=f"localhost:5000/pytorch-cpu:{_IMAGE_TAG}",
    packages_to_install=[],
)
def train_classifier(
    train_dataset: Input[Dataset],
    val_dataset: Input[Dataset],
    num_epochs: int,
    learning_rate: float,
    batch_size: int,
    hidden_dims: str,
    output_model: Output[Model],
    output_metrics: Output[Metrics],
) -> None:
    """
    MLP 분류 모델 학습.

    hidden_dims: 쉼표 구분 문자열, 예: "256,128,64"
    """
    import json
    import os

    import numpy as np
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

    # -- 데이터 로드 --
    X_train = np.load(os.path.join(train_dataset.path, "features.npy"))
    y_train = np.load(os.path.join(train_dataset.path, "labels.npy"))
    X_val   = np.load(os.path.join(val_dataset.path, "features.npy"))
    y_val   = np.load(os.path.join(val_dataset.path, "labels.npy"))

    num_classes = int(y_train.max()) + 1
    input_dim = X_train.shape[1]
    print(f"[train] input_dim={input_dim}, num_classes={num_classes}")
    print(f"[train] train={len(X_train)}, val={len(X_val)}")

    # -- 데이터셋 / DataLoader --
    def to_loader(X, y, shuffle=True):
        dataset = TensorDataset(
            torch.tensor(X, dtype=torch.float32),
            torch.tensor(y, dtype=torch.long),
        )
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    train_loader = to_loader(X_train, y_train, shuffle=True)
    val_loader   = to_loader(X_val, y_val, shuffle=False)

    # -- 모델 정의 --
    dims = [input_dim] + [int(d) for d in hidden_dims.split(",")] + [num_classes]

    layers = []
    for i in range(len(dims) - 1):
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        if i < len(dims) - 2:
            layers.append(nn.BatchNorm1d(dims[i + 1]))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.3))
    model = nn.Sequential(*layers)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    print(f"[train] device={device}, model layers={len(layers)}")

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=5, factor=0.5, verbose=True
    )

    # -- 학습 루프 --
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_val_acc = 0.0

    for epoch in range(1, num_epochs + 1):
        # Train
        model.train()
        t_loss, t_correct = 0.0, 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            t_loss += loss.item() * len(xb)
            t_correct += (logits.argmax(1) == yb).sum().item()

        t_loss /= len(X_train)
        t_acc = t_correct / len(X_train)

        # Validation
        model.eval()
        v_loss, v_correct = 0.0, 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                logits = model(xb)
                v_loss += criterion(logits, yb).item() * len(xb)
                v_correct += (logits.argmax(1) == yb).sum().item()

        v_loss /= len(X_val)
        v_acc = v_correct / len(X_val)
        scheduler.step(v_loss)

        history["train_loss"].append(t_loss)
        history["train_acc"].append(t_acc)
        history["val_loss"].append(v_loss)
        history["val_acc"].append(v_acc)

        if epoch % 10 == 0 or epoch == 1:
            print(
                f"[train] epoch {epoch:03d}/{num_epochs} | "
                f"loss {t_loss:.4f} acc {t_acc:.4f} | "
                f"val_loss {v_loss:.4f} val_acc {v_acc:.4f}"
            )

        # 최고 val_acc 모델 저장
        if v_acc > best_val_acc:
            best_val_acc = v_acc
            os.makedirs(output_model.path, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(output_model.path, "best_model.pt"))

    # -- 모델 메타데이터 저장 --
    model_meta = {
        "input_dim": input_dim,
        "num_classes": num_classes,
        "hidden_dims": hidden_dims,
        "num_epochs": num_epochs,
        "learning_rate": learning_rate,
        "batch_size": batch_size,
    }
    with open(os.path.join(output_model.path, "model_meta.json"), "w") as f:
        json.dump(model_meta, f, indent=2)

    with open(os.path.join(output_model.path, "history.json"), "w") as f:
        json.dump(history, f, indent=2)

    print(f"[train] 학습 완료. best_val_acc={best_val_acc:.4f}")

    # -- KFP 메트릭 --
    output_metrics.log_metric("best_val_accuracy", round(best_val_acc, 4))
    output_metrics.log_metric("final_train_loss", round(history["train_loss"][-1], 4))
    output_metrics.log_metric("final_val_loss", round(history["val_loss"][-1], 4))
    output_metrics.log_metric("num_classes", num_classes)
