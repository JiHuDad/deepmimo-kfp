"""
register_model 컴포넌트 (범용)

MLflow Model Registry 에 모델을 등록하고 선택적으로 스테이지를 승격한다.

전제 조건:
    - train_classifier 컴포넌트가 use_mlflow=True 로 실행되어 있어야 함
    - trained_model 아티팩트 내 mlflow_run_id.txt 파일이 존재해야 함

동작:
    1. mlflow_run_id.txt 에서 run_id 읽기
    2. mlflow.register_model() 으로 Model Registry 등록 (버전 생성)
    3. promote_to_stage != "" 이면 해당 스테이지로 전환 (Staging / Production)

환경 변수 (ConfigMap/Secret 주입):
    MLFLOW_TRACKING_URI      — MLflow 서버 주소
    AWS_ACCESS_KEY_ID        — MinIO 접근 키
    AWS_SECRET_ACCESS_KEY    — MinIO 비밀 키
    MLFLOW_S3_ENDPOINT_URL   — MinIO 엔드포인트 (artifact 다운로드용)
"""

import os

from kfp import dsl
from kfp.dsl import Input, Output, Model, Metrics

_IMAGE_TAG = os.environ.get("IMAGE_TAG", "latest")


@dsl.component(
    base_image=f"localhost:5000/pytorch-cpu:{_IMAGE_TAG}",
    packages_to_install=[],
)
def register_model(
    trained_model: Input[Model],
    output_metrics: Output[Metrics],
    model_name: str = "beam-selection-mlp",
    promote_to_stage: str = "",
    mlflow_tracking_uri: str = "",
) -> None:
    """
    MLflow Model Registry 에 모델 등록.

    model_name: Registry 에 등록할 모델 이름
    promote_to_stage: 등록 후 전환할 스테이지 ("Staging", "Production", "" = 전환 안 함)
    mlflow_tracking_uri: ConfigMap 환경변수(MLFLOW_TRACKING_URI)가 우선 적용됨
    """
    import os

    import mlflow
    from mlflow import MlflowClient

    # -- MLflow URI 설정 --
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "") or mlflow_tracking_uri
    if not tracking_uri:
        raise RuntimeError(
            "MLFLOW_TRACKING_URI 가 설정되지 않았습니다. "
            "ConfigMap 주입 또는 mlflow_tracking_uri 파라미터를 확인하세요."
        )
    mlflow.set_tracking_uri(tracking_uri)
    print(f"[register_model] MLflow URI: {tracking_uri}")

    # -- run_id 읽기 --
    run_id_path = os.path.join(trained_model.path, "mlflow_run_id.txt")
    if not os.path.exists(run_id_path):
        raise FileNotFoundError(
            f"mlflow_run_id.txt 를 찾을 수 없습니다: {run_id_path}\n"
            "train_classifier 가 use_mlflow=True 로 실행되었는지 확인하세요."
        )
    with open(run_id_path) as f:
        run_id = f.read().strip()
    print(f"[register_model] run_id={run_id}")

    # -- Model Registry 등록 --
    model_uri = f"runs:/{run_id}/model"
    mv = mlflow.register_model(model_uri=model_uri, name=model_name)
    version = mv.version
    print(f"[register_model] 등록 완료: {model_name} v{version}")

    # -- 스테이지 전환 --
    if promote_to_stage:
        client = MlflowClient()
        client.transition_model_version_stage(
            name=model_name,
            version=version,
            stage=promote_to_stage,
            archive_existing_versions=True,   # 기존 동일 스테이지 버전 → Archived
        )
        print(f"[register_model] 스테이지 전환: {model_name} v{version} → {promote_to_stage}")

    # -- KFP 메트릭 --
    output_metrics.log_metric("registered_model_name", model_name)
    output_metrics.log_metric("registered_model_version", int(version))
    if promote_to_stage:
        output_metrics.log_metric("promoted_to_stage", promote_to_stage)
