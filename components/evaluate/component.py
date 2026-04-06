"""
evaluate 컴포넌트

학습된 모델로 테스트 세트에서 성능 평가.
- Top-1 / Top-3 Beam Selection 정확도
- Spectral Efficiency (SE) 비율
- 학습 곡선 플롯 생성
"""

from kfp import dsl
from kfp.dsl import Input, Output, Dataset, Model, Metrics, ClassificationMetrics


@dsl.component(
    base_image="localhost:5000/deepmimo-trainer:latest",
    packages_to_install=[],
)
def evaluate(
    test_dataset: Input[Dataset],
    trained_model: Input[Model],
    output_metrics: Output[Metrics],
    output_clf_metrics: Output[ClassificationMetrics],
    output_plots: Output[Dataset],
) -> None:
    """
    테스트 세트 평가 및 시각화.
    """
    import json
    import os

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    import torch
    import torch.nn as nn

    # ── 메타데이터 및 모델 로드 ──────────────────────────────
    meta_path = os.path.join(trained_model.path, "model_meta.json")
    with open(meta_path) as f:
        meta = json.load(f)

    input_dim   = meta["input_dim"]
    num_classes = meta["num_classes"]
    hidden_dims = meta["hidden_dims"]

    dims = [input_dim] + [int(d) for d in hidden_dims.split(",")] + [num_classes]
    layers = []
    for i in range(len(dims) - 1):
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        if i < len(dims) - 2:
            layers.append(nn.BatchNorm1d(dims[i + 1]))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.3))
    model = nn.Sequential(*layers)

    state = torch.load(
        os.path.join(trained_model.path, "best_model.pt"),
        map_location="cpu",
    )
    model.load_state_dict(state)
    model.eval()

    # ── 테스트 데이터 로드 ───────────────────────────────────
    X_test = np.load(os.path.join(test_dataset.path, "features.npy"))
    y_test = np.load(os.path.join(test_dataset.path, "labels.npy"))
    channel_test = np.load(os.path.join(test_dataset.path, "channel.npy"))

    with torch.no_grad():
        logits = model(torch.tensor(X_test, dtype=torch.float32))
        probs  = torch.softmax(logits, dim=1).numpy()

    # ── Top-1 / Top-3 정확도 ─────────────────────────────────
    top1_preds = np.argmax(probs, axis=1)
    top3_preds = np.argsort(probs, axis=1)[:, -3:]

    top1_acc = np.mean(top1_preds == y_test)
    top3_acc = np.mean([y in top3 for y, top3 in zip(y_test, top3_preds)])

    print(f"[evaluate] Top-1 Accuracy: {top1_acc:.4f}")
    print(f"[evaluate] Top-3 Accuracy: {top3_acc:.4f}")

    # ── Spectral Efficiency 비율 ─────────────────────────────
    # SE 비율 = SE(predicted beam) / SE(optimal beam)
    # SE ∝ log2(1 + |h^H * f|^2 / noise) 에서 noise=1로 단순화
    ch = channel_test[:, 0, :, 0]  # (N, n_tx_ant) — 첫 서브캐리어, 첫 RX 안테나
    print(f"[evaluate] ch shape: {ch.shape}, dtype: {ch.dtype}")
    print(f"[evaluate] ch mean abs: {np.abs(ch).mean():.2e}, max: {np.abs(ch).max():.2e}")

    # 채널을 사용자별 RMS로 정규화 (채널 크기의 절대값에 무관하게 SE 비율 계산)
    ch_rms = np.sqrt((np.abs(ch) ** 2).mean(axis=1, keepdims=True)) + 1e-30
    ch_norm = ch / ch_rms  # (N, n_ant), 정규화된 채널

    def beam_se(h_vec, beam_indices):
        """DFT 코드북 기반 빔포밍 SE 계산 (SNR=20dB 가정)."""
        n_ant = h_vec.shape[1]
        snr = 100.0  # 20dB
        beams = np.exp(
            2j * np.pi * np.arange(n_ant)[:, None]
            * np.arange(n_ant)[None, :] / n_ant
        ) / np.sqrt(n_ant)  # (n_ant, n_ant) DFT codebook
        selected_beams = beams[:, beam_indices]  # (n_ant, N)
        bf_gain = np.abs(np.einsum("ni,in->n", h_vec, selected_beams)) ** 2  # (N,)
        return np.log2(1 + snr * bf_gain)

    se_optimal   = beam_se(ch_norm, y_test)
    se_predicted = beam_se(ch_norm, top1_preds)
    se_ratio = np.mean(se_predicted / (se_optimal + 1e-10))
    print(f"[evaluate] SE Ratio: {se_ratio:.4f}")

    # ── KFP 메트릭 ───────────────────────────────────────────
    output_metrics.log_metric("top1_accuracy", round(float(top1_acc), 4))
    output_metrics.log_metric("top3_accuracy", round(float(top3_acc), 4))
    output_metrics.log_metric("se_ratio", round(float(se_ratio), 4))
    output_metrics.log_metric("test_samples", len(y_test))

    # Confusion matrix (상위 10개 클래스만)
    top_classes = list(range(min(10, num_classes)))
    mask = np.isin(y_test, top_classes)
    if mask.sum() > 0:
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(
            y_test[mask], top1_preds[mask], labels=top_classes
        )
        output_clf_metrics.log_confusion_matrix(
            [str(c) for c in top_classes],
            cm.tolist(),
        )

    # ── 학습 곡선 플롯 ───────────────────────────────────────
    history_path = os.path.join(trained_model.path, "history.json")
    with open(history_path) as f:
        history = json.load(f)

    os.makedirs(output_plots.path, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(history["train_loss"], label="Train Loss")
    axes[0].plot(history["val_loss"],   label="Val Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Training & Validation Loss")
    axes[0].legend()
    axes[0].grid(True)

    axes[1].plot(history["train_acc"], label="Train Acc")
    axes[1].plot(history["val_acc"],   label="Val Acc")
    axes[1].axhline(y=top1_acc, color="r", linestyle="--", label=f"Test Top-1: {top1_acc:.3f}")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_title("Training & Validation Accuracy")
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()
    plot_path = os.path.join(output_plots.path, "training_curve.png")
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"[evaluate] 학습 곡선 저장 → {plot_path}")
