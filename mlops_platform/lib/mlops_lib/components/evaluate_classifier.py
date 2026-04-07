"""
evaluate_classifier 컴포넌트 (범용)

학습된 MLP 분류 모델로 테스트 세트에서 성능을 평가한다.
- Top-1 / Top-3 정확도
- Confusion Matrix (상위 10개 클래스)
- 학습 곡선 플롯
- 예측 결과를 아티팩트로 출력 (프로젝트별 도메인 평가에 활용)

입력 데이터셋 형식:
    test_dataset/features.npy  — (N, D) float32
    test_dataset/labels.npy    — (N,)   int64

출력:
    output_predictions/predictions.npy — (N, C) float32 softmax 확률
    output_predictions/labels.npy      — (N,)   int64 실제 레이블
"""

import os

from kfp import dsl
from kfp.dsl import Input, Output, Dataset, Model, Metrics, ClassificationMetrics

_IMAGE_TAG = os.environ.get("IMAGE_TAG", "latest")


@dsl.component(
    base_image=f"localhost:5000/pytorch-cpu:{_IMAGE_TAG}",
    packages_to_install=[],
)
def evaluate_classifier(
    test_dataset: Input[Dataset],
    trained_model: Input[Model],
    output_metrics: Output[Metrics],
    output_clf_metrics: Output[ClassificationMetrics],
    output_predictions: Output[Dataset],
    output_plots: Output[Dataset],
) -> None:
    """
    범용 분류 모델 테스트 세트 평가 및 시각화.
    예측 결과(softmax 확률)를 아티팩트로 출력하여
    프로젝트별 도메인 평가 컴포넌트에서 활용할 수 있도록 한다.
    """
    import json
    import os

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    import torch
    import torch.nn as nn

    # -- 메타데이터 및 모델 로드 --
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

    # -- 테스트 데이터 로드 --
    X_test = np.load(os.path.join(test_dataset.path, "features.npy"))
    y_test = np.load(os.path.join(test_dataset.path, "labels.npy"))

    with torch.no_grad():
        logits = model(torch.tensor(X_test, dtype=torch.float32))
        probs  = torch.softmax(logits, dim=1).numpy()

    # -- Top-1 / Top-3 정확도 --
    top1_preds = np.argmax(probs, axis=1)
    top3_preds = np.argsort(probs, axis=1)[:, -3:]

    top1_acc = np.mean(top1_preds == y_test)
    top3_acc = np.mean([y in top3 for y, top3 in zip(y_test, top3_preds)])

    print(f"[evaluate] Top-1 Accuracy: {top1_acc:.4f}")
    print(f"[evaluate] Top-3 Accuracy: {top3_acc:.4f}")

    # -- KFP 메트릭 --
    output_metrics.log_metric("top1_accuracy", round(float(top1_acc), 4))
    output_metrics.log_metric("top3_accuracy", round(float(top3_acc), 4))
    output_metrics.log_metric("test_samples", len(y_test))

    # -- 예측 결과 저장 (도메인 평가 컴포넌트용) --
    os.makedirs(output_predictions.path, exist_ok=True)
    np.save(os.path.join(output_predictions.path, "predictions.npy"), probs)
    np.save(os.path.join(output_predictions.path, "top1_preds.npy"), top1_preds)
    np.save(os.path.join(output_predictions.path, "labels.npy"), y_test)

    # -- Confusion matrix (상위 10개 클래스) --
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

    # -- 학습 곡선 플롯 --
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
    print(f"[evaluate] 학습 곡선 저장: {plot_path}")
