# Phase 2 — MLflow 모델 레지스트리 통합 (옵션)

> 학습된 모델을 MLflow에 등록하여 버전·Stage·메트릭을 중앙에서 관리한다.
> **하위 호환**: 기존 PVC 저장 방식을 그대로 유지하면서, 옵션으로 MLflow를 활성화한다.

## 목표

1. k3s 폐쇄망에 MLflow Tracking Server 배포
2. `train_classifier`에 `use_mlflow: bool` 파라미터 추가 (기본 False)
3. `mlops_platform/.../components/register_model.py` 신규 컴포넌트 작성
4. 평가/도메인 메트릭(SE 등)을 MLflow Run에 함께 기록
5. 모델 등록 시 `feature_schema_version` 태그 부착 (Phase 1과 연동)
6. 폐쇄망 wheel 추가, Docker base 이미지에 mlflow 포함

## 비목표

- KServe 자동 배포 → Phase 3
- A/B 테스트, Shadow Deploy → 추후
- MLflow Experiments UI 커스터마이징

---

## 배경

현재 학습이 끝나면:
- 모델은 KFP `Output[Model]` → MinIO/PVC에 unique 경로로 저장
- 어떤 모델이 "현재 production"인지 어디에도 표시 없음
- 메트릭은 KFP UI에서만 확인 가능, 비교 도구 빈약
- Phase 3에서 KServe가 모델을 가져갈 명확한 진실의 원천(SoT)이 없음

MLflow Model Registry는 다음을 제공한다:
- 모델 이름 + 버전 (`deepmimo-beam-selector` `v1`, `v2`, ...)
- Stage 전이: None → Staging → Production → Archived
- Run/Experiment 단위 메트릭 추적
- KServe `mlflow://` URI 직접 지원

---

## 아키텍처

```
                                       ┌──────────────────────────┐
                                       │   MLflow Tracking Server │
                                       │   (Pod in k3s)           │
                                       │                          │
                                       │ ┌──────────────────────┐ │
                                       │ │ Backend Store        │ │
                                       │ │  PostgreSQL or SQLite│ │
                                       │ └──────────────────────┘ │
                                       │ ┌──────────────────────┐ │
                                       │ │ Artifact Store       │ │
                                       │ │  MinIO (s3 호환)     │ │
                                       │ └──────────────────────┘ │
                                       └──────────────▲───────────┘
                                                      │
              ┌─────────────────┐                     │
              │ train_classifier├──────log_metrics ───┤
              │   use_mlflow=T  │                     │
              └────────┬────────┘                     │
                       │ output_model (PVC)           │
                       ▼                              │
              ┌─────────────────┐                     │
              │ register_model  ├──── log_model ──────┘
              │   (NEW)         │      register
              └─────────────────┘      transition_stage
```

---

## 인프라 작업: MLflow 서버 배포

### 1) 폐쇄망 wheel 수집

`offline-packages/collect.sh` 에 추가:

```bash
# MLflow + 의존성
pip download \
  mlflow==2.18.0 \
  mlflow-skinny==2.18.0 \
  alembic sqlalchemy pyarrow boto3 \
  -d offline-packages/wheels/
```

### 2) MLflow Docker 이미지

`mlops_platform/base-images/mlflow-server/Dockerfile`:

```dockerfile
ARG IMAGE_TAG=latest
FROM localhost:5000/python-cpu:${IMAGE_TAG}

COPY offline-packages/wheels/ /tmp/wheels/
RUN pip install --no-index --find-links=/tmp/wheels/ \
        mlflow==2.18.0 boto3 psycopg2-binary \
    && rm -rf /tmp/wheels/

EXPOSE 5000
CMD ["mlflow", "server", \
     "--host", "0.0.0.0", \
     "--port", "5000", \
     "--backend-store-uri", "sqlite:////mlflow/mlflow.db", \
     "--default-artifact-root", "s3://mlflow-artifacts/"]
```

> SQLite는 단일 사용자/소규모용. 운영 시 PostgreSQL로 교체.

### 3) Kubernetes 매니페스트

`mlops_platform/k8s/mlflow/`:

```
mlflow-server-deployment.yaml    # MLflow Deployment + Service
mlflow-pvc.yaml                  # SQLite DB용 PVC (1Gi)
minio-deployment.yaml            # MinIO (artifact store)
minio-pvc.yaml                   # 아티팩트 저장용 (50Gi)
```

설치 스크립트: `mlops_platform/scripts/setup-mlflow.sh`

```bash
#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/lib/common.sh"

log_info "MLflow + MinIO 배포 중..."
kubectl apply -f mlops_platform/k8s/mlflow/

log_info "Pod ready 대기..."
kubectl wait --for=condition=ready pod -l app=mlflow --timeout=300s
kubectl wait --for=condition=ready pod -l app=minio  --timeout=300s

log_ok "MLflow UI: http://${SERVER_IP}:31500"
log_ok "MinIO UI:  http://${SERVER_IP}:31900"
```

---

## 컴포넌트 변경

### 1) `train_classifier` — MLflow 옵션 추가

```python
# mlops_platform/lib/mlops_lib/components/train_classifier.py

@dsl.component(base_image=f"localhost:5000/pytorch-cpu:{_IMAGE_TAG}")
def train_classifier(
    train_dataset: Input[Dataset],
    val_dataset: Input[Dataset],
    num_epochs: int,
    learning_rate: float,
    batch_size: int,
    hidden_dims: str,
    output_model: Output[Model],
    output_metrics: Output[Metrics],
    # ── NEW ─────────────────────────────────────────
    use_mlflow: bool = False,
    mlflow_tracking_uri: str = "",
    mlflow_experiment: str = "default",
    mlflow_run_name: str = "",
) -> None:
    import os, json
    import numpy as np
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

    # ... 기존 학습 코드 ...

    # ── MLflow 로깅 (옵션) ─────────────────────────
    if use_mlflow:
        import mlflow
        mlflow.set_tracking_uri(mlflow_tracking_uri)
        mlflow.set_experiment(mlflow_experiment)
        with mlflow.start_run(run_name=mlflow_run_name or None) as run:
            mlflow.log_params({
                "num_epochs": num_epochs,
                "learning_rate": learning_rate,
                "batch_size": batch_size,
                "hidden_dims": hidden_dims,
                "input_dim": input_dim,
                "num_classes": num_classes,
            })
            for ep, (tl, ta, vl, va) in enumerate(zip(
                history["train_loss"], history["train_acc"],
                history["val_loss"],   history["val_acc"]), start=1):
                mlflow.log_metrics({
                    "train_loss": tl, "train_acc": ta,
                    "val_loss":   vl, "val_acc":   va,
                }, step=ep)
            mlflow.log_metric("best_val_accuracy", best_val_acc)
            mlflow.pytorch.log_model(model, artifact_path="model")
            # Run ID를 메타에 저장 → register_model 컴포넌트가 사용
            with open(os.path.join(output_model.path, "mlflow_run_id"), "w") as f:
                f.write(run.info.run_id)
```

### 2) `register_model` — 신규 컴포넌트

```python
# mlops_platform/lib/mlops_lib/components/register_model.py

"""
register_model 컴포넌트 (범용)

학습된 모델을 MLflow Model Registry에 등록하고 stage를 전환한다.

입력:
    trained_model: train_classifier 출력
                   - best_model.pt
                   - model_meta.json
                   - mlflow_run_id (use_mlflow=True 였을 때만)

동작:
    1. mlflow_run_id 를 읽어 해당 run의 model artifact를 등록
    2. registered_model_name 으로 새 버전 생성
    3. promote_to_stage 가 지정되면 transition (Staging / Production)
    4. tags 에 feature_schema_version 등 부착
"""

import os
from kfp import dsl
from kfp.dsl import Input, Model, Metrics

_IMAGE_TAG = os.environ.get("IMAGE_TAG", "latest")


@dsl.component(
    base_image=f"localhost:5000/pytorch-cpu:{_IMAGE_TAG}",
    packages_to_install=[],
)
def register_model(
    trained_model: Input[Model],
    output_metrics: dsl.Output[Metrics],
    mlflow_tracking_uri: str,
    registered_model_name: str,
    promote_to_stage: str = "",
    tags: str = "{}",
) -> None:
    import json
    import mlflow
    from mlflow.tracking import MlflowClient

    mlflow.set_tracking_uri(mlflow_tracking_uri)
    client = MlflowClient()

    run_id_path = os.path.join(trained_model.path, "mlflow_run_id")
    if not os.path.exists(run_id_path):
        raise RuntimeError(
            "mlflow_run_id 파일이 없습니다. "
            "train_classifier에서 use_mlflow=True 로 실행했는지 확인하세요."
        )

    with open(run_id_path) as f:
        run_id = f.read().strip()

    model_uri = f"runs:/{run_id}/model"
    print(f"[register_model] 등록: {model_uri} → {registered_model_name}")

    mv = mlflow.register_model(model_uri, registered_model_name)
    print(f"[register_model] 등록 완료: version={mv.version}")

    # 태그 부착
    parsed_tags = json.loads(tags) if tags else {}
    meta_path = os.path.join(trained_model.path, "model_meta.json")
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            meta = json.load(f)
        for key in ("input_dim", "num_classes", "feature_schema_version"):
            if key in meta:
                parsed_tags.setdefault(key, str(meta[key]))

    for k, v in parsed_tags.items():
        client.set_model_version_tag(registered_model_name, mv.version, k, v)

    # Stage 전환
    if promote_to_stage:
        client.transition_model_version_stage(
            name=registered_model_name,
            version=mv.version,
            stage=promote_to_stage,
            archive_existing_versions=(promote_to_stage == "Production"),
        )
        print(f"[register_model] stage={promote_to_stage}")

    output_metrics.log_metric("registered_version", int(mv.version))
```

### 3) 파이프라인 연결

`projects/deepmimo_beam_selection/pipeline.py` (요지):

```python
def deepmimo_pipeline(
    # 기존 파라미터들...
    use_mlflow: bool = False,
    mlflow_tracking_uri: str = "http://mlflow.mlops.svc.cluster.local:5000",
    promote_to_stage: str = "",          # "" / "Staging" / "Production"
):
    # ... validate, preprocess ...

    train_task = train_classifier(
        train_dataset=preprocess_task.outputs["output_train"],
        val_dataset=preprocess_task.outputs["output_val"],
        num_epochs=num_epochs,
        learning_rate=learning_rate,
        batch_size=batch_size,
        hidden_dims=hidden_dims,
        use_mlflow=use_mlflow,
        mlflow_tracking_uri=mlflow_tracking_uri,
        mlflow_experiment="deepmimo-beam-selection",
    )

    eval_task = evaluate_classifier(...)
    se_task   = evaluate_se(...)

    # ── MLflow 사용 시에만 실행 ─────────────────────
    with dsl.If(use_mlflow == True):
        register_task = register_model(
            trained_model=train_task.outputs["output_model"],
            mlflow_tracking_uri=mlflow_tracking_uri,
            registered_model_name="deepmimo-beam-selector",
            promote_to_stage=promote_to_stage,
        )
        register_task.after(eval_task, se_task)
```

> 주의: KFP v2에서 `dsl.If` 조건은 컴파일 시 평가되지 않고
> 런타임 파라미터를 비교한다. SDK 버전에 따라 문법 차이가 있으므로
> 실제 작성 시 `kfp.dsl.Condition` API를 확인할 것.

---

## 평가 컴포넌트 정리 (Phase 1 후속)

현재 `evaluate_classifier`(플랫폼)과 `evaluate_se`(프로젝트)로 나뉘어 있다.
Phase 2를 진행하면서 다음과 같이 정리하면 깔끔하다.

### 옵션 A — 현재 구조 유지 (권장)

플랫폼 `evaluate_classifier`은 그대로 두고, 도메인 메트릭은
`projects/.../components/evaluate_se.py`에서 별도 컴포넌트로 실행.

장점: 분리 명확
단점: 컴포넌트 수 +1

### 옵션 B — 도메인 평가 컴포넌트 통합

`projects/.../components/evaluate.py` 신규로 만들어
정확도, SE, 도메인 지표를 한 컴포넌트에서 계산하고 MLflow에 기록.

```python
# projects/deepmimo_beam_selection/components/evaluate.py
@dsl.component(base_image=f"localhost:5000/deepmimo-base:{_IMAGE_TAG}")
def evaluate(
    test_dataset: Input[Dataset],
    trained_model: Input[Model],
    output_metrics: Output[Metrics],
    use_mlflow: bool = False,
    mlflow_tracking_uri: str = "",
):
    # 1. Top-1 / Top-3 (범용)
    # 2. SE ratio (도메인 특화)
    # 3. (옵션) MLflow run에 추가 메트릭 로깅
```

장점: 모델 개발자가 평가를 한 곳에서 관리
단점: 플랫폼 컴포넌트 일부 중복

> 선택은 Phase 2 PR 시점에 팀 합의로 결정.
> 본 문서는 옵션 A를 기준으로 작성.

---

## 작업 순서

1. **인프라**
   - [ ] `offline-packages/collect.sh` 에 mlflow wheel 추가
   - [ ] MLflow Docker 이미지 빌드 스크립트
   - [ ] k8s 매니페스트 작성 (mlflow + minio)
   - [ ] `setup-mlflow.sh` 작성 + 동작 확인
   - [ ] MLflow UI 접근 확인 (`http://${SERVER_IP}:31500`)

2. **컴포넌트**
   - [ ] `train_classifier` MLflow 옵션 추가
   - [ ] `register_model.py` 신규 작성
   - [ ] `mlops_lib/components/__init__.py` 에 export 등록
   - [ ] pytorch-cpu base 이미지에 `mlflow` 추가
   - [ ] 이미지 재빌드

3. **파이프라인**
   - [ ] `pipeline.py` 에 `use_mlflow`, `promote_to_stage` 파라미터 추가
   - [ ] `register_model` 태스크 조건부 연결
   - [ ] `compile.py` 갱신

4. **검증**
   - [ ] `use_mlflow=False` (기존 동작) 회귀 테스트
   - [ ] `use_mlflow=True, promote_to_stage=""` 실행
     - MLflow UI에서 Run/Model 확인
   - [ ] `use_mlflow=True, promote_to_stage="Staging"` 실행
     - Model Registry에서 stage=Staging 확인
   - [ ] `promote_to_stage="Production"` 실행
     - 이전 Production 자동 archive 확인

5. **문서**
   - [ ] README에 MLflow UI 접근 방법 추가
   - [ ] Phase 3 사전 조건 확인 (`mlflow://` URI 동작)

---

## 검증 체크리스트

- [ ] MLflow Pod, MinIO Pod 가 Running
- [ ] `use_mlflow=False` 일 때 기존과 동일하게 동작 (회귀 없음)
- [ ] `use_mlflow=True` 일 때 Run에 params/metrics/artifacts 모두 기록
- [ ] `register_model` 컴포넌트가 새 버전 생성 + 태그 부착
- [ ] Stage 전환이 정상 동작
- [ ] `feature_schema_version` 태그가 모델 버전에 붙어 있음
- [ ] MLflow UI에서 검색/비교 가능

---

## 위험 요소

| 위험 | 완화 방안 |
|------|-----------|
| 폐쇄망 wheel 누락 | `offline-packages/` 의존성 trial-run 후 확정 |
| MLflow + MinIO 보안 (인증 없음) | 내부망 ClusterIP만 노출, NodePort는 신뢰망에서만 접근 |
| KFP 컴포넌트에서 MLflow tracking 실패 시 학습 전체 실패 | `try/except`로 감싸 학습 결과는 유지하고 경고 로그만 출력 |
| Run ID 전달 메커니즘 (파일 기반) | 명시적 contract로 문서화 → register_model이 검증 |
| MLflow 버전 호환 (모델 포맷) | 학습/등록/서빙이 동일한 mlflow 버전을 사용 — 이미지 태그로 강제 |

---

## 다음 단계

Phase 2 머지 후, MLflow Production stage에 모델이 1개 이상 등록된 상태에서
`03-phase3-kserve.md` 검토 → KServe로 자동 배포.
