# Phase 3 — KServe 서빙 배포

> MLflow에 Production 단계로 등록된 모델을 KServe로 자동 배포한다.
> Phase 1의 `features/extractor.py`를 KServe Transformer가 그대로 import하여
> Training-Serving Skew를 0으로 만든다.

## 목표

1. k3s 폐쇄망에 KServe 설치
2. `projects/.../serving/transformer.py` 작성 — `features/extractor.py` 재사용
3. `projects/.../serving/inference_service.yaml` 작성
4. MLflow에서 모델 자동 pull (`storageUri: mlflow://...`)
5. HTTP 예측 엔드포인트 동작 확인 + curl 예시
6. 모델 업데이트 시 자동 롤링 배포 흐름 정의

## 비목표

- A/B 테스트, Canary, Shadow → 기본 배포 후 별도 단계로
- gRPC 인터페이스 → REST 먼저
- 멀티 모델 (ensemble) → 단일 모델 먼저
- 오토스케일링 튜닝 → 기본값으로 시작

---

## 사전 조건

- Phase 1 머지 (`features/extractor.py` 존재 + 동작)
- Phase 2 머지 (MLflow Production stage에 모델 1개 이상 존재)
- k3s 클러스터에 충분한 리소스 (Predictor 1Gi + Transformer 512Mi 정도)

---

## KServe 개념 빠르게 이해하기

```
HTTP Request
    │
    ▼
┌───────────────────────────┐
│  InferenceService         │  ← 사용자가 정의하는 K8s 리소스 (CRD)
└─────────────┬─────────────┘
              │
    ┌─────────┴─────────┐
    ▼                   ▼
┌──────────┐      ┌──────────────┐
│Transformer│ ──► │  Predictor   │
│  Pod     │      │     Pod      │
│          │      │              │
│ 전처리    │      │ 모델 추론    │
│ 후처리    │      │ (PyTorch/    │
│          │      │  Triton/     │
│          │      │  MLflow)     │
└──────────┘      └──────────────┘
```

- **Predictor**는 모델 가중치를 로드해 forward만 한다 (도메인 무관)
- **Transformer**는 raw 요청을 모델 입력으로, 모델 출력을 응답으로 가공
- 두 Pod는 별도 컨테이너 — Transformer가 실패해도 Predictor는 살아있다

> KServe v0.11+ 는 MLflow Model Registry URI를 직접 지원한다:
> `storageUri: "mlflow://<registered_model_name>/<stage_or_version>"`

---

## 인프라 작업: KServe 설치

폐쇄망 환경에서는 다음 이미지를 미리 USB로 가져와 로컬 레지스트리에 푸시한다.

```
kserve/kserve-controller
kserve/agent
kserve/storage-initializer
kserve/mlflowserver           ← MLflow 모델 로딩 담당
knative/activator             ← Knative serving (KServe 의존성)
knative/autoscaler
istio/proxyv2                 ← (옵션) Istio 사용 시
```

### 설치 스크립트

`mlops_platform/scripts/setup-kserve.sh`:

```bash
#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/lib/common.sh"

log_info "Knative Serving 설치..."
kubectl apply -f mlops_platform/k8s/kserve/knative-serving-crds.yaml
kubectl apply -f mlops_platform/k8s/kserve/knative-serving-core.yaml

log_info "KServe CRD + 컨트롤러 설치..."
kubectl apply -f mlops_platform/k8s/kserve/kserve.yaml

log_info "ClusterServingRuntime 등록..."
kubectl apply -f mlops_platform/k8s/kserve/cluster-serving-runtimes.yaml

log_info "Pod ready 대기..."
kubectl -n kserve wait --for=condition=ready pod -l control-plane=kserve-controller-manager --timeout=300s

log_ok "KServe 설치 완료"
```

> 폐쇄망 manifest는 KServe 공식 release YAML에서 image 경로를
> `localhost:5000/...` 로 일괄 치환한 사본을 사용한다.

---

## 디렉토리 구조

```
projects/deepmimo_beam_selection/
├── features/
│   └── extractor.py            ← Phase 1 결과
├── serving/                    ← NEW (Phase 3)
│   ├── __init__.py
│   ├── transformer.py          ← KServe Transformer 진입점
│   ├── Dockerfile              ← Transformer 컨테이너 이미지
│   ├── requirements.txt
│   ├── inference_service.yaml  ← KServe IS 정의
│   └── examples/
│       ├── request.json        ← 예시 요청
│       └── curl.sh             ← curl 호출 예시
└── ...
```

---

## 코드 설계

### 1) `serving/transformer.py`

KServe Transformer는 `kserve.Model` 을 상속한 클래스를 만들고
`preprocess()`, `postprocess()` 를 정의한다.

```python
"""
DeepMIMO Beam Selection — KServe Transformer

사용자 요청 (raw 채널) → feature 벡터로 변환 → Predictor 호출
Predictor 응답 (logits) → top-1 빔 인덱스 + 확률

학습 파이프라인의 features/extractor.py 와 동일 모듈을 import 하여
Training-Serving Skew를 방지한다.
"""

from __future__ import annotations

import argparse
import logging
from typing import Any

import numpy as np
import kserve
from kserve import InferRequest, InferResponse, Model, ModelServer

# 학습 시 사용한 것과 동일한 모듈
from projects.deepmimo_beam_selection.features import (
    extract_features,
    schema as feature_schema,
)

logger = logging.getLogger(__name__)


class BeamSelectionTransformer(Model):
    """
    Request body 형식 (JSON):
        {
          "instances": [
            {
              "channels": <list of complex, shape (rx, tx, subc)>,
              "channels_real": [...],         # 또는 real/imag 분리 전송
              "channels_imag": [...]
            }
          ]
        }

    Response body:
        {
          "predictions": [
            {"top1_beam": 17, "probabilities": [0.1, 0.05, ...]}
          ]
        }
    """

    def __init__(self, name: str, predictor_host: str):
        super().__init__(name)
        self.predictor_host = predictor_host
        self.ready = False
        self._schema = feature_schema()
        logger.info(f"Transformer 시작 — feature_schema={self._schema}")

    def load(self) -> bool:
        # Predictor 헬스체크는 KServe 프레임워크가 자동 처리
        self.ready = True
        return True

    # ── 전처리 ─────────────────────────────────────────────
    def preprocess(self, payload: dict, headers: dict | None = None) -> dict:
        instances = payload.get("instances", [])
        if not instances:
            raise ValueError("instances가 비어있습니다")

        # 각 instance의 channels를 numpy로 변환
        channels_list = []
        for inst in instances:
            real = np.asarray(inst["channels_real"], dtype=np.float32)
            imag = np.asarray(inst["channels_imag"], dtype=np.float32)
            ch = real + 1j * imag                # (rx, tx, subc)
            channels_list.append(ch)

        # (N, rx, tx, subc) 형태로 stack
        channels = np.stack(channels_list, axis=0)

        # ★ 학습과 동일한 함수 사용 ★
        features = extract_features(channels)    # (N, 2*tx)

        return {"instances": features.tolist()}

    # ── 후처리 ─────────────────────────────────────────────
    def postprocess(self, response: dict, headers: dict | None = None) -> dict:
        # Predictor 응답: {"predictions": [[...logits...], ...]}
        logits_list = response.get("predictions", [])
        outputs = []
        for logits in logits_list:
            arr = np.asarray(logits, dtype=np.float32)
            probs = _softmax(arr)
            outputs.append({
                "top1_beam": int(np.argmax(arr)),
                "probabilities": probs.tolist(),
                "feature_schema_version": self._schema["version"],
            })
        return {"predictions": outputs}


def _softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - np.max(x))
    return e / e.sum()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default="deepmimo-beam-selector")
    parser.add_argument("--predictor_host", required=True)
    args, _ = parser.parse_known_args()

    transformer = BeamSelectionTransformer(args.model_name, args.predictor_host)
    transformer.load()
    ModelServer().start([transformer])
```

### 2) `serving/Dockerfile`

```dockerfile
ARG IMAGE_TAG=latest
FROM localhost:5000/deepmimo-base:${IMAGE_TAG}

# KServe + 의존성
COPY offline-packages/wheels/ /tmp/wheels/
COPY projects/deepmimo_beam_selection/serving/requirements.txt /tmp/req.txt
RUN pip install --no-index --find-links=/tmp/wheels/ -r /tmp/req.txt \
    && rm -rf /tmp/wheels/ /tmp/req.txt

# features/ 모듈은 이미 deepmimo-base에 포함되어 있다고 가정 (Phase 1 변경)
# transformer 코드 복사
COPY projects/deepmimo_beam_selection/serving/ /app/projects/deepmimo_beam_selection/serving/

ENV PYTHONPATH=/app
EXPOSE 8080

ENTRYPOINT ["python", "-m", "projects.deepmimo_beam_selection.serving.transformer"]
```

### 3) `serving/requirements.txt`

```
kserve==0.13.0
ray[serve]==2.10.0     # KServe transformer 의존성
```

> 정확한 버전은 `offline-packages/wheels/` 에서 사용 가능한 것으로 결정.

### 4) `serving/inference_service.yaml`

```yaml
apiVersion: serving.kserve.io/v1beta1
kind: InferenceService
metadata:
  name: deepmimo-beam-selector
  namespace: kserve-models
  annotations:
    serving.kserve.io/deploymentMode: "RawDeployment"   # 폐쇄망/단순화
spec:
  predictor:
    minReplicas: 1
    maxReplicas: 2
    model:
      modelFormat:
        name: mlflow
      storageUri: "mlflow://deepmimo-beam-selector/Production"
      resources:
        requests:
          cpu: "500m"
          memory: "1Gi"
        limits:
          cpu: "2"
          memory: "2Gi"

  transformer:
    minReplicas: 1
    maxReplicas: 2
    containers:
      - name: kserve-container
        image: localhost:5000/deepmimo-transformer:latest
        env:
          - name: STORAGE_URI
            value: "mlflow://deepmimo-beam-selector/Production"
          - name: PREDICTOR_HOST
            value: "deepmimo-beam-selector-predictor.kserve-models.svc.cluster.local"
        resources:
          requests:
            cpu: "200m"
            memory: "512Mi"
          limits:
            cpu: "1"
            memory: "1Gi"
```

> `storageUri: mlflow://...` 가 동작하려면 KServe `mlflowserver`
> ClusterServingRuntime이 사전에 설치되어 있어야 한다 (위 setup 스크립트).

### 5) MLflow 인증 정보

KServe Predictor가 MLflow에서 모델을 pull하려면 MLflow Tracking URI와
MinIO 자격증명을 알아야 한다. Secret으로 주입한다.

```yaml
# k8s/kserve/mlflow-credentials-secret.yaml
apiVersion: v1
kind: Secret
metadata:
  name: mlflow-creds
  namespace: kserve-models
type: Opaque
stringData:
  MLFLOW_TRACKING_URI: "http://mlflow.mlops.svc.cluster.local:5000"
  MLFLOW_S3_ENDPOINT_URL: "http://minio.mlops.svc.cluster.local:9000"
  AWS_ACCESS_KEY_ID: "minioadmin"
  AWS_SECRET_ACCESS_KEY: "minioadmin"
```

`InferenceService.spec.predictor.serviceAccountName` 또는 envFrom으로 연결.

---

## 추론 호출 예시

### `examples/request.json`

```json
{
  "instances": [
    {
      "channels_real": [[[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]]],
      "channels_imag": [[[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]]]
    }
  ]
}
```

> shape = (1, 1, 8, 1) — N=1 user, rx=1, tx=8, subc=1

### `examples/curl.sh`

```bash
#!/usr/bin/env bash
SERVICE_URL=$(kubectl get inferenceservice deepmimo-beam-selector \
  -n kserve-models -o jsonpath='{.status.url}')

curl -v -H "Content-Type: application/json" \
     -d @projects/deepmimo_beam_selection/serving/examples/request.json \
     "${SERVICE_URL}/v1/models/deepmimo-beam-selector:predict"
```

기대 응답:

```json
{
  "predictions": [
    {
      "top1_beam": 7,
      "probabilities": [0.01, 0.02, 0.05, ..., 0.42],
      "feature_schema_version": "1.0.0"
    }
  ]
}
```

---

## 모델 업데이트 흐름

```
1. 새 학습 파이프라인 실행
     use_mlflow=True
     promote_to_stage="Staging"

2. 사람이 staging 모델 검증 (smoke test)

3. 만족하면 stage 전환
     mlflow models transition-stage \
       --name deepmimo-beam-selector \
       --version <new> --stage Production

4. KServe Predictor가 다음 reconcile 시점에 새 모델 자동 pull
   (또는 InferenceService를 한 번 patch하여 강제 reload)

5. Transformer는 변경 없음 (feature 로직이 그대로면)
   feature 로직이 바뀌었다면 Transformer 이미지도 재빌드 + rollout
```

> **Feature 로직이 바뀐 경우**: Transformer와 학습 모델을 동시에 업데이트해야 한다.
> `feature_schema_version` 태그로 두 컴포넌트의 호환성을 표시한다.
> 응답에 `feature_schema_version`을 포함시켜 클라이언트도 검증 가능하게 한다.

---

## 작업 순서

1. **인프라**
   - [ ] KServe + Knative 이미지 폐쇄망 USB 수집
   - [ ] 로컬 레지스트리 푸시
   - [ ] `mlops_platform/k8s/kserve/` 매니페스트 작성 (이미지 경로 치환)
   - [ ] `setup-kserve.sh` 작성
   - [ ] 설치 후 `kubectl get pods -n kserve` 확인

2. **Transformer**
   - [ ] `serving/transformer.py` 작성
   - [ ] `serving/Dockerfile` + `requirements.txt`
   - [ ] `kserve` 등 wheel 수집 + `python-cpu`/`deepmimo-base`에 포함
   - [ ] 이미지 빌드: `localhost:5000/deepmimo-transformer:latest`
   - [ ] 이미지 푸시

3. **InferenceService**
   - [ ] `serving/inference_service.yaml` 작성
   - [ ] `mlflow-creds` Secret 생성
   - [ ] 네임스페이스 생성: `kubectl create ns kserve-models`
   - [ ] `kubectl apply -f inference_service.yaml`
   - [ ] `kubectl get inferenceservice` 에서 READY=True 확인

4. **검증**
   - [ ] curl 호출 시 200 OK
   - [ ] `top1_beam` 결과가 학습 시 분포와 유사
   - [ ] Transformer pod 로그에 feature_schema_version 출력 확인
   - [ ] 학습 batch eval 결과와 서빙 결과 일치 (sanity check)

5. **운영 준비**
   - [ ] MLflow에서 stage transition 시 자동 reload 확인 (또는 수동 patch)
   - [ ] Transformer rollout 절차 문서화
   - [ ] 장애 시 롤백 절차 문서화 (이전 버전으로 stage 전환)

---

## 검증 체크리스트

- [ ] KServe / Knative Pod 모두 Running
- [ ] InferenceService READY=True
- [ ] curl 요청 → 200 응답
- [ ] 응답의 `top1_beam` 이 정상 범위 (0..n_tx-1)
- [ ] Transformer 로그에 feature_schema_version 기록
- [ ] 학습 시점과 동일 입력에 동일 출력 (offline 검증)
- [ ] MLflow stage 전환 후 새 버전이 서빙되는지 확인
- [ ] Transformer Pod 재시작 시 자동 복구

---

## 위험 요소

| 위험 | 완화 방안 |
|------|-----------|
| 폐쇄망 KServe 이미지 누락 | 사전 USB 수집 + 체크리스트 점검 |
| KServe `mlflow://` URI 미지원 (구 버전) | 0.11 이상 강제 — `mlflowserver` ClusterServingRuntime 확인 |
| Training-Serving Skew | features/ 모듈을 양쪽이 동일 import — 응답에 schema version 포함하여 검증 |
| Transformer ↔ Predictor 통신 실패 | KServe가 자동 헬스체크/재시작 — 그래도 안 되면 같은 Pod에 sidecar로 |
| 모델 업데이트 시 다운타임 | minReplicas=2, RollingUpdate, Knative blue/green |
| MinIO 자격증명 노출 | Secret으로 주입, RBAC로 namespace 제한 |
| 응답 latency 큰 경우 | RawDeployment 사용 (Knative cold-start 회피) |

---

## 향후 확장

- **Canary 배포**: KServe `canaryTrafficPercent` 로 신모델에 트래픽 일부 라우팅
- **Shadow 배포**: 기존 모델과 신모델에 동일 요청 전송, 결과 비교 로깅
- **Multi-model serving**: ModelMesh로 여러 모델을 단일 Pod에 호스팅
- **gRPC**: 저지연 클라이언트 대응
- **Drift Monitoring**: 입력 분포 모니터링 → 재학습 트리거

---

## 작업량 추정

- KServe 폐쇄망 설치: 1~2일 (이미지 수집/조정 시간)
- Transformer 코드 + 이미지: 0.5~1일
- 인증 / Secret / 네트워크: 0.5일
- 검증 / 디버깅: 1~2일
- 총: **3~5일** (폐쇄망 환경의 변수 큼)

---

## 완료 후 상태

- 학습 파이프라인 → MLflow 등록 → Production 전환 → KServe 자동 서빙
- 모델 개발자는 `features/extractor.py`, `model 코드`, `evaluate.py` 만 관리
- MLOps는 플랫폼/Transformer/InferenceService 만 관리
- HTTP `/predict` 엔드포인트로 실시간 빔 선택 추론 가능
- 모든 단계가 Git + MLflow에서 추적 가능
