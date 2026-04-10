"""
deepmimo_beam_selection 파이프라인 (DeepMIMO v4)

DeepMIMO v4 API 기반 빔 선택 모델 학습 파이프라인.
폐쇄망 환경 기준: 시나리오 데이터는 deepmimo-scenarios PVC에 사전 적재.

플랫폼 범용 컴포넌트 + 프로젝트 전용 컴포넌트 조합:
  validate_data (플랫폼) → preprocess (프로젝트)
    → train_classifier (플랫폼) → evaluate_classifier (플랫폼)
    → evaluate_se (프로젝트)
    → [선택] register_model (플랫폼, use_mlflow=True 일 때만 실행)

MLflow 연동 (use_mlflow=True):
  - train_classifier: 실험 추적, 모델 아티팩트 로깅
  - register_model:   Model Registry 등록 및 스테이지 승격
  - ConfigMap(mlflow-config), Secret(mlflow-s3-creds) 는 항상 주입됨
    (컴포넌트가 use_mlflow=False 이면 환경변수를 무시함)

주의: use_mlflow=True 이면 클러스터에 아래 리소스가 필요합니다.
  kubectl apply -f mlops_platform/k8s/mlflow/configmap.yaml
  kubectl apply -f mlops_platform/k8s/mlflow/secret.yaml
"""

import sys
from pathlib import Path

# 프로젝트 루트를 sys.path에 추가
_project_root = str(Path(__file__).parent.parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from kfp import dsl, kubernetes

from mlops_platform.lib.mlops_lib.components import (
    validate_data,
    train_classifier,
    evaluate_classifier,
    register_model,
)

from projects.deepmimo_beam_selection.components import preprocess, evaluate_se


def _inject_mlflow_env(task):
    """ConfigMap + Secret 을 태스크 환경변수로 주입한다."""
    kubernetes.use_config_map_as_env(
        task,
        config_map_name="mlflow-config",
        config_map_key_to_env={
            "MLFLOW_TRACKING_URI":    "MLFLOW_TRACKING_URI",
            "MLFLOW_S3_ENDPOINT_URL": "MLFLOW_S3_ENDPOINT_URL",
        },
    )
    kubernetes.use_secret_as_env(
        task,
        secret_name="mlflow-s3-creds",
        secret_key_to_env={
            "AWS_ACCESS_KEY_ID":     "AWS_ACCESS_KEY_ID",
            "AWS_SECRET_ACCESS_KEY": "AWS_SECRET_ACCESS_KEY",
        },
    )


@dsl.pipeline(
    name="deepmimo-beam-selection",
    description="DeepMIMO v4 레이트레이싱 데이터 기반 빔 선택 파이프라인 (폐쇄망)",
)
def deepmimo_pipeline(
    # 시나리오
    scenario_name: str = "asu_campus_3p5",
    scenario_source_path: str = "/data/scenarios",
    # DeepMIMO v4 채널 파라미터
    bs_antenna_shape: str = "8,1",       # BS 안테나 배열 (n_h, n_v)
    num_subcarriers: int = 512,
    bandwidth: float = 20.0,             # MHz (asu_campus_3p5: 3.5 GHz 대역)
    num_paths: int = 5,                  # 로드할 최대 경로 수 (0 = DeepMIMO 기본값 10)
    # DeepMIMO TX/RX set 선택
    tx_set_id: int = 0,            # BS TX set 인덱스 (asu_campus_3p5 기본값)
    rx_set_id: int = 0,            # UE RX set 인덱스
    max_users: int = 50000,        # 0 = 전체 사용자 (메모리 주의)
    # 데이터 분할
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    random_seed: int = 42,         # 데이터 분할 재현성
    # 학습 하이퍼파라미터
    num_epochs: int = 50,
    learning_rate: float = 1e-3,
    batch_size: int = 256,
    hidden_dims: str = "256,128,64",
    # MLflow 연동
    use_mlflow: bool = False,
    mlflow_experiment_name: str = "asu-campus-beam-selection",
    mlflow_model_name: str = "asu-campus-beam-selection-mlp",
    promote_to_stage: str = "",          # "Staging" | "Production" | "" (등록만)
) -> None:

    # ── Step 1: 데이터 검증 (플랫폼 범용) ───────────────────
    validate_task = validate_data(
        data_name=scenario_name,
        data_source_path=scenario_source_path,
    )
    kubernetes.mount_pvc(
        validate_task,
        pvc_name="deepmimo-scenarios",
        mount_path="/data/scenarios",
    )
    validate_task.set_display_name("1. 데이터 검증")

    # ── Step 2: 채널 생성 및 분할 (프로젝트 전용) ────────────
    preprocess_task = preprocess(
        scenario_dataset=validate_task.outputs["output_dataset"],
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
        random_seed=random_seed,
    )
    kubernetes.mount_pvc(
        preprocess_task,
        pvc_name="deepmimo-scenarios",
        mount_path="/data/scenarios",
    )
    preprocess_task.set_cpu_request("2")
    preprocess_task.set_memory_request("4Gi")
    preprocess_task.set_display_name("2. 채널 생성 및 분할")

    # ── Step 3: 학습 (플랫폼 범용) ───────────────────────────
    # ConfigMap + Secret 은 항상 주입됨.
    # use_mlflow=False 이면 컴포넌트 내부에서 환경변수를 무시한다.
    # (ConfigMap/Secret 이 클러스터에 없으면 Pod 기동 실패 → 아래 setup-mlflow.sh 참조)
    train_task = train_classifier(
        train_dataset=preprocess_task.outputs["output_train"],
        val_dataset=preprocess_task.outputs["output_val"],
        num_epochs=num_epochs,
        learning_rate=learning_rate,
        batch_size=batch_size,
        hidden_dims=hidden_dims,
        use_mlflow=use_mlflow,
        mlflow_experiment_name=mlflow_experiment_name,
    )
    _inject_mlflow_env(train_task)
    train_task.set_cpu_request("4")
    train_task.set_memory_request("8Gi")
    train_task.set_display_name("3. 모델 학습")

    # ── Step 4: 분류 성능 평가 (플랫폼 범용) ─────────────────
    eval_task = evaluate_classifier(
        test_dataset=preprocess_task.outputs["output_test"],
        trained_model=train_task.outputs["output_model"],
    )
    eval_task.set_display_name("4. 분류 성능 평가")

    # ── Step 5: SE 평가 (프로젝트 전용) ──────────────────────
    se_task = evaluate_se(
        test_dataset=preprocess_task.outputs["output_test"],
        predictions=eval_task.outputs["output_predictions"],
    )
    se_task.set_display_name("5. Spectral Efficiency 평가")

    # ── Step 6: Model Registry 등록 (MLflow 옵션) ────────────
    # use_mlflow=True 일 때만 실행
    with dsl.If(use_mlflow == True, name="register-if-mlflow"):  # noqa: E712
        register_task = register_model(
            trained_model=train_task.outputs["output_model"],
            model_name=mlflow_model_name,
            promote_to_stage=promote_to_stage,
        )
        _inject_mlflow_env(register_task)
        register_task.set_display_name("6. 모델 레지스트리 등록")
