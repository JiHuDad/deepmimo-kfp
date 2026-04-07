"""
[MLOps 제공] Pipeline Factory

create_pipeline(model_type) 한 줄로 임의 모델의 전체 파이프라인을 생성한다.
모델 개발자는 이 파일을 수정할 필요가 없습니다.

사용 예:
    from framework.pipeline_factory import create_pipeline

    # 단일 모델 파이프라인
    pipeline = create_pipeline("mlp")

    # 여러 모델 비교 파이프라인
    pipeline = create_multi_pipeline(["mlp", "transformer"])
"""

from __future__ import annotations

import importlib
import json

from kfp import compiler, dsl, kubernetes

from components.load_scenario.component import load_scenario
from components.preprocess.component import preprocess
from components.evaluate.component import evaluate
from framework.train_component import train_generic


def _load_default_config(model_type: str) -> dict:
    """models/<model_type>/model.py 의 Model.default_config() 를 읽어온다."""
    try:
        module = importlib.import_module(f"models.{model_type}.model")
        return module.Model.default_config()
    except (ModuleNotFoundError, AttributeError):
        return {}


def create_pipeline(model_type: str):
    """
    단일 모델 타입에 대한 전체 KFP 파이프라인을 생성하여 반환한다.

    Args:
        model_type: models/ 디렉토리 안에 있는 모델 이름.
                    예: "mlp", "transformer"

    Returns:
        @dsl.pipeline 이 적용된 파이프라인 함수.
        kfp.compiler.Compiler().compile() 또는 KFP client에 직접 전달 가능.

    사용 예:
        pipeline = create_pipeline("mlp")
        compiler.Compiler().compile(pipeline, "mlp_pipeline.yaml")
    """
    default_config = _load_default_config(model_type)
    default_config_str = json.dumps(default_config)

    @dsl.pipeline(
        name=f"deepmimo-{model_type}",
        description=f"DeepMIMO 빔 선택 파이프라인 — 모델: {model_type}",
    )
    def _pipeline(
        # ── 시나리오 파라미터 ──────────────────────────────────
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
        # ── 공통 학습 파라미터 ─────────────────────────────────
        num_epochs: int = 50,
        batch_size: int = 256,
        # ── 모델별 하이퍼파라미터 (JSON 문자열) ────────────────
        # default_config()에서 자동으로 기본값을 가져온다
        model_config: str = default_config_str,
    ) -> None:

        # Step 1: 시나리오 로드
        load_task = load_scenario(
            scenario_name=scenario_name,
            scenario_source_path=scenario_source_path,
        )
        kubernetes.mount_pvc(
            load_task, pvc_name="deepmimo-scenarios", mount_path="/data/scenarios"
        )
        load_task.set_display_name("1. 시나리오 로드")

        # Step 2: 전처리
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
        kubernetes.mount_pvc(
            preprocess_task, pvc_name="deepmimo-scenarios", mount_path="/data/scenarios"
        )
        preprocess_task.set_cpu_request("2").set_memory_request("4Gi")
        preprocess_task.set_display_name("2. 전처리")

        # Step 3: 학습 (모델 타입은 이미지 안에서 동적 로드)
        train_task = train_generic(
            train_dataset=preprocess_task.outputs["output_train"],
            val_dataset=preprocess_task.outputs["output_val"],
            model_type=model_type,      # 컴파일 시 문자열로 고정
            model_config=model_config,  # 런타임에 변경 가능
            num_epochs=num_epochs,
            batch_size=batch_size,
        )
        train_task.set_cpu_request("4").set_memory_request("8Gi")
        train_task.set_display_name(f"3. 학습 ({model_type})")

        # Step 4: 평가
        evaluate_task = evaluate(
            test_dataset=preprocess_task.outputs["output_test"],
            trained_model=train_task.outputs["output_model"],
        )
        evaluate_task.set_display_name("4. 평가")

    # 파이프라인 함수에 이름을 붙여 디버깅 편의성 향상
    _pipeline.__name__ = f"deepmimo_{model_type}_pipeline"
    return _pipeline


def create_multi_pipeline(model_types: list[str]):
    """
    여러 모델을 동일 데이터로 병렬 학습하는 비교 파이프라인을 생성한다.

    전처리는 1번만 실행되고, 학습/평가는 각 모델마다 병렬로 실행된다.

    Args:
        model_types: 비교할 모델 이름 목록.  예: ["mlp", "transformer"]

    사용 예:
        pipeline = create_multi_pipeline(["mlp", "transformer"])
        compiler.Compiler().compile(pipeline, "compare_pipeline.yaml")
    """
    model_names = "_vs_".join(model_types)
    default_configs = {m: _load_default_config(m) for m in model_types}

    # 각 모델의 config 기본값을 파이프라인 파라미터로 펼침
    # 예: mlp_config='{"hidden_dims":"256,128,64"}', transformer_config='{"d_model":64}'
    default_config_strs = {m: json.dumps(default_configs[m]) for m in model_types}

    @dsl.pipeline(
        name=f"deepmimo-compare-{model_names}",
        description=f"DeepMIMO 모델 비교: {', '.join(model_types)}",
    )
    def _multi_pipeline(
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
        # 각 모델의 config를 독립 파라미터로 노출
        # KFP UI에서 모델별로 따로 조정 가능
        **{f"{m}_config": default_config_strs[m] for m in model_types},
    ) -> None:

        # Step 1: 시나리오 로드
        load_task = load_scenario(
            scenario_name=scenario_name,
            scenario_source_path=scenario_source_path,
        )
        kubernetes.mount_pvc(
            load_task, pvc_name="deepmimo-scenarios", mount_path="/data/scenarios"
        )
        load_task.set_display_name("1. 시나리오 로드")

        # Step 2: 전처리 (공통 — 1번만 실행)
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
        kubernetes.mount_pvc(
            preprocess_task, pvc_name="deepmimo-scenarios", mount_path="/data/scenarios"
        )
        preprocess_task.set_cpu_request("2").set_memory_request("4Gi")
        preprocess_task.set_display_name("2. 전처리 (공통)")

        # Step 3~4: 각 모델별 학습 + 평가 (병렬)
        for model_type in model_types:
            train_task = train_generic(
                train_dataset=preprocess_task.outputs["output_train"],
                val_dataset=preprocess_task.outputs["output_val"],
                model_type=model_type,
                model_config=default_config_strs[model_type],
                num_epochs=num_epochs,
                batch_size=batch_size,
            )
            train_task.set_cpu_request("4").set_memory_request("8Gi")
            train_task.set_display_name(f"3. 학습 ({model_type})")

            evaluate_task = evaluate(
                test_dataset=preprocess_task.outputs["output_test"],
                trained_model=train_task.outputs["output_model"],
            )
            evaluate_task.set_display_name(f"4. 평가 ({model_type})")

    _multi_pipeline.__name__ = f"deepmimo_compare_{model_names}_pipeline"
    return _multi_pipeline


def compile_pipeline(model_type: str, output_path: str | None = None) -> str:
    """
    파이프라인을 YAML로 컴파일하여 저장한다.

    Args:
        model_type : 모델 이름
        output_path: 저장 경로 (None이면 자동 생성)

    Returns:
        저장된 YAML 파일 경로
    """
    if output_path is None:
        output_path = f"deepmimo_{model_type}_pipeline.yaml"
    pipeline = create_pipeline(model_type)
    compiler.Compiler().compile(pipeline, output_path)
    print(f"컴파일 완료: {output_path}")
    return output_path
