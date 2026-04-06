"""
deepmimo_beam_selection 파이프라인 (DeepMIMO v4)

DeepMIMO v4 API 기반 빔 선택 모델 학습 파이프라인.
폐쇄망 환경 기준: 시나리오 데이터는 deepmimo-scenarios PVC에 사전 적재.

파이프라인 순서:
  load_scenario → preprocess → train → evaluate
"""

from kfp import compiler, dsl, kubernetes

from components.load_scenario.component import load_scenario
from components.preprocess.component import preprocess
from components.train.component import train
from components.evaluate.component import evaluate


@dsl.pipeline(
    name="deepmimo-beam-selection",
    description="DeepMIMO v4 레이트레이싱 데이터 기반 빔 선택 파이프라인 (폐쇄망)",
)
def deepmimo_pipeline(
    # 시나리오
    scenario_name: str = "O1_60",
    scenario_source_path: str = "/data/scenarios",
    # DeepMIMO v4 채널 파라미터
    bs_antenna_shape: str = "8,1",       # BS 안테나 배열 (n_h, n_v)
    num_subcarriers: int = 512,
    bandwidth: float = 50.0,             # MHz
    num_paths: int = 5,                  # 0 = 전체 경로
    # DeepMIMO TX/RX set 선택
    tx_set_id: int = 3,            # BS TX set 인덱스
    rx_set_id: int = 0,            # UE RX set 인덱스
    max_users: int = 50000,        # 0 = 전체 사용자 (메모리 주의)
    # 데이터 분할
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    # 학습 하이퍼파라미터
    num_epochs: int = 50,
    learning_rate: float = 1e-3,
    batch_size: int = 256,
    hidden_dims: str = "256,128,64",
) -> None:

    # ── Step 1: 시나리오 로드 ────────────────────────────────
    load_task = load_scenario(
        scenario_name=scenario_name,
        scenario_source_path=scenario_source_path,
    )
    kubernetes.mount_pvc(
        load_task,
        pvc_name="deepmimo-scenarios",
        mount_path="/data/scenarios",
    )
    load_task.set_display_name("1. 시나리오 로드")

    # ── Step 2: 채널 생성 및 분할 ────────────────────────────
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
    # 시나리오 데이터를 PVC에서 직접 읽기 위해 마운트 (복사 없음)
    kubernetes.mount_pvc(
        preprocess_task,
        pvc_name="deepmimo-scenarios",
        mount_path="/data/scenarios",
    )
    preprocess_task.set_cpu_request("2")
    preprocess_task.set_memory_request("4Gi")
    preprocess_task.set_display_name("2. 채널 생성 및 분할")

    # ── Step 3: 학습 ─────────────────────────────────────────
    train_task = train(
        train_dataset=preprocess_task.outputs["output_train"],
        val_dataset=preprocess_task.outputs["output_val"],
        num_epochs=num_epochs,
        learning_rate=learning_rate,
        batch_size=batch_size,
        hidden_dims=hidden_dims,
    )
    train_task.set_cpu_request("4")
    train_task.set_memory_request("8Gi")
    train_task.set_display_name("3. 모델 학습")

    # ── Step 4: 평가 ─────────────────────────────────────────
    evaluate_task = evaluate(
        test_dataset=preprocess_task.outputs["output_test"],
        trained_model=train_task.outputs["output_model"],
    )
    evaluate_task.set_display_name("4. 성능 평가")
