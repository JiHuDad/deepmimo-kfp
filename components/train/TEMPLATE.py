"""
[MLOps 제공] 학습 컴포넌트 템플릿
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
사용법:
  1. 이 파일을 복사해서 models/<모델이름>/component.py 로 저장
  2. ✏️  표시된 3개 구간만 채우면 됩니다
  3. 나머지는 수정하지 마세요

모델 개발자가 작성할 부분:
  - [1] 모델 클래스 정의    (build_model 함수)
  - [2] 옵티마이저 설정     (build_optimizer 함수)
  - [3] 추가 하이퍼파라미터 (component 파라미터)

MLOps가 관리하는 부분 (수정 금지):
  - 데이터 로드 / DataLoader 생성
  - 학습 루프 (train/val 루프, best model 저장)
  - 아티팩트 저장 (model_meta.json, history.json)
  - KFP 메트릭 로깅
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

from kfp import dsl
from kfp.dsl import Input, Output, Dataset, Model, Metrics


@dsl.component(
    base_image="localhost:5000/deepmimo-trainer:latest",
    packages_to_install=[
        # ✏️ [선택] 추가 패키지가 필요하면 여기에 입력
        # 예: "timm==0.9.12", "einops==0.7.0"
    ],
)
def train(
    train_dataset: Input[Dataset],
    val_dataset: Input[Dataset],
    # ── [MLOps 공통 파라미터] ──────────────────────────────
    num_epochs: int,
    learning_rate: float,
    batch_size: int,
    # ✏️ [2] 모델별 추가 하이퍼파라미터를 여기에 선언
    # 예: hidden_dims: str = "256,128,64",
    # 예: num_heads: int = 4,
    # 예: num_layers: int = 3,
    # ──────────────────────────────────────────────────────
    output_model: Output[Model],
    output_metrics: Output[Metrics],
) -> None:
    import json
    import os

    import numpy as np
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

    # ════════════════════════════════════════════════════════
    # [MLOps 관리] 데이터 로드  ← 수정 금지
    # ════════════════════════════════════════════════════════
    X_train = np.load(os.path.join(train_dataset.path, "features.npy"))
    y_train = np.load(os.path.join(train_dataset.path, "labels.npy"))
    X_val   = np.load(os.path.join(val_dataset.path,   "features.npy"))
    y_val   = np.load(os.path.join(val_dataset.path,   "labels.npy"))

    num_classes = int(y_train.max()) + 1
    input_dim   = X_train.shape[1]
    print(f"[train] input_dim={input_dim}, num_classes={num_classes}")
    print(f"[train] train={len(X_train)}, val={len(X_val)}")

    def to_loader(X, y, shuffle=True):
        ds = TensorDataset(
            torch.tensor(X, dtype=torch.float32),
            torch.tensor(y, dtype=torch.long),
        )
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)

    train_loader = to_loader(X_train, y_train, shuffle=True)
    val_loader   = to_loader(X_val,   y_val,   shuffle=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[train] device={device}")

    # ════════════════════════════════════════════════════════
    # ✏️ [1] 모델 정의  ← 여기를 채우세요
    # ════════════════════════════════════════════════════════
    def build_model(input_dim: int, num_classes: int) -> nn.Module:
        """
        모델 아키텍처를 정의하고 nn.Module을 반환하세요.
        input_dim  : 입력 피처 차원 (데이터에서 자동 결정)
        num_classes: 출력 클래스 수 (데이터에서 자동 결정)
        """
        raise NotImplementedError("build_model() 을 구현해주세요")

    # ════════════════════════════════════════════════════════
    # ✏️ [2] 옵티마이저 & 스케줄러 설정  ← 여기를 채우세요
    # ════════════════════════════════════════════════════════
    def build_optimizer(model: nn.Module):
        """
        (optimizer, scheduler) 튜플을 반환하세요.
        scheduler가 필요 없으면 (optimizer, None) 반환.
        """
        raise NotImplementedError("build_optimizer() 를 구현해주세요")

    # ════════════════════════════════════════════════════════
    # [MLOps 관리] 학습 루프  ← 수정 금지
    # ════════════════════════════════════════════════════════
    model = build_model(input_dim, num_classes).to(device)
    optimizer, scheduler = build_optimizer(model)
    criterion = nn.CrossEntropyLoss()

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_val_acc = 0.0

    for epoch in range(1, num_epochs + 1):
        model.train()
        t_loss, t_correct = 0.0, 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            t_loss    += loss.item() * len(xb)
            t_correct += (model(xb).argmax(1) == yb).sum().item()
        t_loss /= len(X_train)
        t_acc   = t_correct / len(X_train)

        model.eval()
        v_loss, v_correct = 0.0, 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                logits  = model(xb)
                v_loss    += criterion(logits, yb).item() * len(xb)
                v_correct += (logits.argmax(1) == yb).sum().item()
        v_loss /= len(X_val)
        v_acc   = v_correct / len(X_val)

        if scheduler is not None:
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

        if v_acc > best_val_acc:
            best_val_acc = v_acc
            os.makedirs(output_model.path, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(output_model.path, "best_model.pt"))

    # ════════════════════════════════════════════════════════
    # [MLOps 관리] 저장 & 메트릭  ← 수정 금지
    # ════════════════════════════════════════════════════════
    # ✏️ [3] model_meta에 모델별 파라미터를 추가하고 싶으면
    #        아래 딕셔너리에 키를 추가하세요
    model_meta = {
        "input_dim": input_dim,
        "num_classes": num_classes,
        "num_epochs": num_epochs,
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        # 예: "hidden_dims": hidden_dims,
        # 예: "num_heads": num_heads,
    }
    with open(os.path.join(output_model.path, "model_meta.json"), "w") as f:
        json.dump(model_meta, f, indent=2)
    with open(os.path.join(output_model.path, "history.json"), "w") as f:
        json.dump(history, f, indent=2)

    output_metrics.log_metric("best_val_accuracy", round(best_val_acc, 4))
    output_metrics.log_metric("final_train_loss",  round(history["train_loss"][-1], 4))
    output_metrics.log_metric("final_val_loss",    round(history["val_loss"][-1], 4))
    output_metrics.log_metric("num_classes", num_classes)
    print(f"[train] 완료. best_val_acc={best_val_acc:.4f}")
