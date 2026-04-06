"""
deepmimo_beam_selection 파이프라인

DeepMIMO 레이트레이싱 채널 데이터를 이용한 빔 선택 모델 학습 파이프라인.
폐쇄망 환경 기준: 시나리오 데이터는 PVC에 사전 적재되어 있어야 한다.

파이프라인 순서:
  load_scenario → preprocess → train → evaluate
"""

import json

from kfp import compiler, dsl, kubernetes

from components.load_scenario.component import load_scenario
from components.preprocess.component import preprocess
from components.train.component import train
from components.evaluate.component import evaluate

# ── 기본 DeepMIMO 파라미터 ─────────────────────────────────────────────────
DEFAULT_DEEPMIMO_PARAMS = json.dumps({
    "num_paths": 5,
    "active_BS": [1],
    "user_row_first": 1,
    "user_row_last": 100,
    "subcarriers": 512,
    "bandwidth": 0.5,
    "num_OFDM_subcarriers": 512,
    "OFDM_limit": 32,
})


@dsl.pipeline(
    name="deepmimo-beam-selection",
    description="DeepMIMO 레이트레이싱 데이터 기반 빔 선택 파이프라인 (폐쇄망)",
)
def deepmimo_pipeline(
    # 시나리오 설정
    scenario_name: str = "O1_60",
    scenario_source_path: str = "/data/scenarios",
    deepmimo_params: str = DEFAULT_DEEPMIMO_PARAMS,
    # 데이터 분할 비율
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
    # PVC 마운트: 시나리오 데이터가 사전 적재된 deepmimo-scenarios PVC
    kubernetes.mount_pvc(
        load_task,
        pvc_name="deepmimo-scenarios",
        mount_path="/data/scenarios",
    )
    load_task.set_display_name("1. 시나리오 로드")

    # ── Step 2: 전처리 ───────────────────────────────────────
    preprocess_task = preprocess(
        scenario_dataset=load_task.outputs["output_scenario"],
        parameters_json=deepmimo_params,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
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
