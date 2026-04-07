"""
[MLOps 제공] Generic Train Component

model_type 문자열을 받아 models/<model_type>/model.py 의 Model 클래스를
동적으로 로드하고 학습을 실행하는 단일 KFP 컴포넌트.

모델 개발자는 이 파일을 수정할 필요가 없습니다.
"""

from kfp import dsl
from kfp.dsl import Input, Output, Dataset, Model, Metrics


@dsl.component(
    base_image="localhost:5000/deepmimo-trainer:latest",
    packages_to_install=[],
)
def train_generic(
    train_dataset: Input[Dataset],
    val_dataset: Input[Dataset],
    model_type: str,        # 예: "mlp", "transformer"  ← 런타임 파라미터
    model_config: str,      # JSON 문자열  예: '{"hidden_dims":"256,128,64"}'
    num_epochs: int,
    batch_size: int,
    output_model: Output[Model],
    output_metrics: Output[Metrics],
) -> None:
    """
    공통 학습 컴포넌트.

    model_type에 해당하는 models/<model_type>/model.py 를 동적 임포트하여
    build_model() / build_optimizer() 를 호출한다.
    학습 루프, 저장, 메트릭 로깅은 모두 이 함수가 처리한다.
    """
    import importlib
    import json
    import os

    import numpy as np
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

    # ── 모델 클래스 동적 로드 ────────────────────────────────────
    # models/<model_type>/model.py 에서 Model 클래스를 가져온다
    try:
        module = importlib.import_module(f"models.{model_type}.model")
    except ModuleNotFoundError:
        raise ValueError(
            f"모델 '{model_type}'을 찾을 수 없습니다. "
            f"models/{model_type}/model.py 가 존재하는지 확인하세요."
        )

    config = json.loads(model_config)
    model_instance = module.Model(config)
    print(f"[train] model_type={model_type}, config={config}")

    # ── 데이터 로드 ─────────────────────────────────────────────
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

    # ── 모델 / 옵티마이저 초기화 ─────────────────────────────────
    model = model_instance.build_model(input_dim, num_classes).to(device)
    optimizer, scheduler = model_instance.build_optimizer(model)
    criterion = nn.CrossEntropyLoss()

    # ── 학습 루프 ────────────────────────────────────────────────
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_val_acc = 0.0

    for epoch in range(1, num_epochs + 1):
        model.train()
        t_loss, t_correct = 0.0, 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            t_loss    += loss.item() * len(xb)
            t_correct += (logits.argmax(1) == yb).sum().item()
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

        # scheduler.step() 호출 방식은 타입에 따라 분기
        if scheduler is not None:
            if hasattr(scheduler, "step"):
                import inspect
                sig = inspect.signature(scheduler.step)
                if "metrics" in sig.parameters:
                    scheduler.step(v_loss)   # ReduceLROnPlateau
                else:
                    scheduler.step()          # CosineAnnealing 등

        history["train_loss"].append(t_loss)
        history["train_acc"].append(t_acc)
        history["val_loss"].append(v_loss)
        history["val_acc"].append(v_acc)

        if epoch % 10 == 0 or epoch == 1:
            print(
                f"[{model_type}] epoch {epoch:03d}/{num_epochs} | "
                f"loss {t_loss:.4f} acc {t_acc:.4f} | "
                f"val_loss {v_loss:.4f} val_acc {v_acc:.4f}"
            )

        if v_acc > best_val_acc:
            best_val_acc = v_acc
            os.makedirs(output_model.path, exist_ok=True)
            torch.save(
                model.state_dict(),
                os.path.join(output_model.path, "best_model.pt"),
            )

    # ── 아티팩트 저장 ────────────────────────────────────────────
    model_meta = {
        "model_type": model_type,
        "input_dim": input_dim,
        "num_classes": num_classes,
        "num_epochs": num_epochs,
        "batch_size": batch_size,
        **config,                         # 모델별 하이퍼파라미터
        **model_instance.extra_meta,      # 모델이 추가로 기록하고 싶은 정보
    }
    with open(os.path.join(output_model.path, "model_meta.json"), "w") as f:
        json.dump(model_meta, f, indent=2)
    with open(os.path.join(output_model.path, "history.json"), "w") as f:
        json.dump(history, f, indent=2)

    output_metrics.log_metric("best_val_accuracy", round(best_val_acc, 4))
    output_metrics.log_metric("final_train_loss",  round(history["train_loss"][-1], 4))
    output_metrics.log_metric("final_val_loss",    round(history["val_loss"][-1], 4))
    output_metrics.log_metric("num_classes", num_classes)
    print(f"[{model_type}] 완료. best_val_acc={best_val_acc:.4f}")
