"""
[모델 개발자 B 작성] Transformer 학습 컴포넌트
TEMPLATE.py 기반 - build_model / build_optimizer 만 채움

MLP와 달리 시퀀스 입력으로 reshape해서 사용.
입력 피처를 (seq_len, d_model) 형태로 변환 후 TransformerEncoder 통과.
"""

from kfp import dsl
from kfp.dsl import Input, Output, Dataset, Model, Metrics


@dsl.component(
    base_image="localhost:5000/deepmimo-trainer:latest",
    packages_to_install=[],  # PyTorch는 base image에 포함
)
def train_transformer(
    train_dataset: Input[Dataset],
    val_dataset: Input[Dataset],
    # ── 공통 파라미터 ──
    num_epochs: int,
    learning_rate: float,
    batch_size: int,
    # ── Transformer 전용 파라미터 ──────────── ✏️ 추가
    d_model: int = 64,       # Transformer 임베딩 차원
    num_heads: int = 4,      # Multi-head attention 헤드 수
    num_layers: int = 2,     # TransformerEncoder 레이어 수
    dropout: float = 0.1,
    # ──────────────────────────────────────────────
    output_model: Output[Model],
    output_metrics: Output[Metrics],
) -> None:
    import json, os, math
    import numpy as np
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

    # ════════════════════════════════════════════════════════
    # [MLOps 관리] 데이터 로드 & DataLoader
    # ════════════════════════════════════════════════════════
    X_train = np.load(os.path.join(train_dataset.path, "features.npy"))
    y_train = np.load(os.path.join(train_dataset.path, "labels.npy"))
    X_val   = np.load(os.path.join(val_dataset.path,   "features.npy"))
    y_val   = np.load(os.path.join(val_dataset.path,   "labels.npy"))

    num_classes = int(y_train.max()) + 1
    input_dim   = X_train.shape[1]
    print(f"[Transformer] input_dim={input_dim}, num_classes={num_classes}")

    def to_loader(X, y, shuffle=True):
        ds = TensorDataset(torch.tensor(X, dtype=torch.float32),
                           torch.tensor(y, dtype=torch.long))
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)

    train_loader = to_loader(X_train, y_train, shuffle=True)
    val_loader   = to_loader(X_val,   y_val,   shuffle=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ════════════════════════════════════════════════════════
    # ✏️ [1] 모델 정의 - 모델 개발자 B 작성
    # ════════════════════════════════════════════════════════
    def build_model(input_dim: int, num_classes: int) -> nn.Module:
        # 입력 피처를 (seq_len, d_model) 시퀀스로 변환
        # input_dim을 d_model로 나눌 수 있도록 seq_len 계산
        seq_len = math.ceil(input_dim / d_model)
        padded_dim = seq_len * d_model  # 패딩 후 총 차원

        class BeamTransformer(nn.Module):
            def __init__(self):
                super().__init__()
                self.pad_dim   = padded_dim
                self.input_proj = nn.Linear(input_dim, padded_dim)
                encoder_layer  = nn.TransformerEncoderLayer(
                    d_model=d_model,
                    nhead=num_heads,
                    dim_feedforward=d_model * 4,
                    dropout=dropout,
                    batch_first=True,   # (batch, seq, d_model)
                )
                self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
                self.classifier = nn.Sequential(
                    nn.LayerNorm(d_model),
                    nn.Linear(d_model, num_classes),
                )

            def forward(self, x):
                x = self.input_proj(x)                   # (B, padded_dim)
                x = x.view(x.size(0), seq_len, d_model)  # (B, seq_len, d_model)
                x = self.encoder(x)                       # (B, seq_len, d_model)
                x = x.mean(dim=1)                         # (B, d_model) - mean pooling
                return self.classifier(x)                 # (B, num_classes)

        return BeamTransformer()

    # ════════════════════════════════════════════════════════
    # ✏️ [2] 옵티마이저 - 모델 개발자 B 작성
    # ════════════════════════════════════════════════════════
    def build_optimizer(model: nn.Module):
        # Transformer에는 AdamW + CosineAnnealing이 더 잘 맞음
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=learning_rate, weight_decay=1e-4
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=num_epochs
        )
        return optimizer, scheduler

    # ════════════════════════════════════════════════════════
    # [MLOps 관리] 학습 루프 (TEMPLATE과 동일)
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
        scheduler.step()  # CosineAnnealing은 인자 없이 호출

        history["train_loss"].append(t_loss)
        history["train_acc"].append(t_acc)
        history["val_loss"].append(v_loss)
        history["val_acc"].append(v_acc)

        if epoch % 10 == 0 or epoch == 1:
            print(f"[Transformer] epoch {epoch:03d}/{num_epochs} | "
                  f"loss {t_loss:.4f} acc {t_acc:.4f} | "
                  f"val_loss {v_loss:.4f} val_acc {v_acc:.4f}")

        if v_acc > best_val_acc:
            best_val_acc = v_acc
            os.makedirs(output_model.path, exist_ok=True)
            torch.save(model.state_dict(),
                       os.path.join(output_model.path, "best_model.pt"))

    # ════════════════════════════════════════════════════════
    # [MLOps 관리] 저장 & 메트릭
    # ════════════════════════════════════════════════════════
    model_meta = {
        "model_type": "transformer",
        "input_dim": input_dim,
        "num_classes": num_classes,
        "d_model": d_model,       # ✏️ Transformer 전용 파라미터 추가
        "num_heads": num_heads,
        "num_layers": num_layers,
        "dropout": dropout,
        "num_epochs": num_epochs,
        "learning_rate": learning_rate,
        "batch_size": batch_size,
    }
    with open(os.path.join(output_model.path, "model_meta.json"), "w") as f:
        json.dump(model_meta, f, indent=2)
    with open(os.path.join(output_model.path, "history.json"), "w") as f:
        json.dump(history, f, indent=2)

    output_metrics.log_metric("best_val_accuracy", round(best_val_acc, 4))
    output_metrics.log_metric("final_train_loss",  round(history["train_loss"][-1], 4))
    output_metrics.log_metric("final_val_loss",    round(history["val_loss"][-1], 4))
    output_metrics.log_metric("num_classes", num_classes)
    print(f"[Transformer] 완료. best_val_acc={best_val_acc:.4f}")
