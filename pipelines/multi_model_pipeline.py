"""
멀티 모델 비교 파이프라인

MLP와 Transformer를 동일한 데이터로 병렬 학습하여 성능을 비교합니다.
전처리(preprocess)는 1번만 실행되고, 학습은 동시에 진행됩니다.

파이프라인 구조:
  load_scenario
      ↓
  preprocess
    ↙       ↘
train_mlp   train_transformer    ← 병렬 실행
    ↓              ↓
evaluate_mlp  evaluate_transformer
"""

from kfp import compiler, dsl, kubernetes

from components.load_scenario.component import load_scenario
from components.preprocess.component import preprocess
from components.evaluate.component import evaluate
from components.train.models.mlp.component import train_mlp
from components.train.models.transformer.component import train_transformer


@dsl.pipeline(
    name="deepmimo-multi-model-comparison",
    description="MLP vs Transformer 빔 선택 모델 비교",
)
def multi_model_pipeline(
    # ── 공통 파라미터 ──────────────────────────────────────
    scenario_name: str = "O1_60",
    scenario_source_path: str = "/data/scenarios",
    bs_antenna_shape: str = "8,1",
    num_subcarriers: int = 512,
    bandwidth: float = 50.0,
    num_paths: int = 5,
    tx_set_id: int = 3,
    rx_set_id: int = 0,
    max_users: int = 50000,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    num_epochs: int = 50,
    batch_size: int = 256,
    # ── 모델별 하이퍼파라미터 ──────────────────────────────
    # MLP
    mlp_learning_rate: float = 1e-3,
    mlp_hidden_dims: str = "256,128,64",
    mlp_dropout: float = 0.3,
    # Transformer
    transformer_learning_rate: float = 5e-4,
    transformer_d_model: int = 64,
    transformer_num_heads: int = 4,
    transformer_num_layers: int = 2,
    transformer_dropout: float = 0.1,
) -> None:

    # ── Step 1: 시나리오 로드 ────────────────────────────────
    load_task = load_scenario(
        scenario_name=scenario_name,
        scenario_source_path=scenario_source_path,
    )
    kubernetes.mount_pvc(load_task, pvc_name="deepmimo-scenarios", mount_path="/data/scenarios")
    load_task.set_display_name("1. 시나리오 로드")

    # ── Step 2: 전처리 (공통, 1번만 실행) ────────────────────
    preprocess_task = preprocess(
        scenario_dataset=load_task.outputs["output_scenario"],
        scenario_name=scenario_name,
        bs_antenna_shape=bs_antenna_shape,
        num_subcarriers=num_subcarriers,
        bandwidth=bandwidth,
        num_paths=num_paths,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        tx_set_id=tx_set_id,
        rx_set_id=rx_set_id,
        max_users=max_users,
    )
    kubernetes.mount_pvc(preprocess_task, pvc_name="deepmimo-scenarios", mount_path="/data/scenarios")
    preprocess_task.set_cpu_request("2").set_memory_request("4Gi")
    preprocess_task.set_display_name("2. 전처리 (공통)")

    # ── Step 3-A: MLP 학습 ───────────────────────────────────
    train_mlp_task = train_mlp(
        train_dataset=preprocess_task.outputs["output_train"],
        val_dataset=preprocess_task.outputs["output_val"],
        num_epochs=num_epochs,
        learning_rate=mlp_learning_rate,
        batch_size=batch_size,
        hidden_dims=mlp_hidden_dims,
        dropout=mlp_dropout,
    )
    train_mlp_task.set_cpu_request("4").set_memory_request("8Gi")
    train_mlp_task.set_display_name("3-A. MLP 학습")

    # ── Step 3-B: Transformer 학습 (MLP와 병렬) ──────────────
    train_transformer_task = train_transformer(
        train_dataset=preprocess_task.outputs["output_train"],
        val_dataset=preprocess_task.outputs["output_val"],
        num_epochs=num_epochs,
        learning_rate=transformer_learning_rate,
        batch_size=batch_size,
        d_model=transformer_d_model,
        num_heads=transformer_num_heads,
        num_layers=transformer_num_layers,
        dropout=transformer_dropout,
    )
    train_transformer_task.set_cpu_request("4").set_memory_request("8Gi")
    train_transformer_task.set_display_name("3-B. Transformer 학습")

    # ── Step 4-A: MLP 평가 ───────────────────────────────────
    evaluate_mlp_task = evaluate(
        test_dataset=preprocess_task.outputs["output_test"],
        trained_model=train_mlp_task.outputs["output_model"],
    )
    evaluate_mlp_task.set_display_name("4-A. MLP 평가")

    # ── Step 4-B: Transformer 평가 ───────────────────────────
    evaluate_transformer_task = evaluate(
        test_dataset=preprocess_task.outputs["output_test"],
        trained_model=train_transformer_task.outputs["output_model"],
    )
    evaluate_transformer_task.set_display_name("4-B. Transformer 평가")


if __name__ == "__main__":
    compiler.Compiler().compile(
        pipeline_func=multi_model_pipeline,
        package_path="multi_model_pipeline.yaml",
    )
    print("컴파일 완료: multi_model_pipeline.yaml")
