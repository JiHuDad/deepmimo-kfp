# MLOps 플랫폼 확장 로드맵

> deepmimo-kfp를 학습-전용 파이프라인에서 **학습 → 등록 → 서빙**까지 책임지는
> End-to-End MLOps 플랫폼으로 확장하기 위한 단계별 계획.

## 목차

- [동기 (Why)](#동기-why)
- [목표 아키텍처](#목표-아키텍처)
- [현재 상태와의 격차](#현재-상태와의-격차)
- [3단계 로드맵 요약](#3단계-로드맵-요약)
- [역할 분리 원칙](#역할-분리-원칙)
- [용어 정리](#용어-정리)

---

## 동기 (Why)

현재 파이프라인은 **학습까지만** 책임지는 구조다. 다음 문제를 해결하기 위해
플랫폼을 단계적으로 확장한다.

| # | 문제 | 영향 |
|---|------|------|
| 1 | feature 추출 로직이 `preprocess.py`에 직접 구현되어 있음 | 서빙 시 동일 로직을 다시 구현해야 함 → **Training-Serving Skew** 위험 |
| 2 | 학습된 모델이 PVC 파일로만 저장됨 | 버전 관리 / Stage(Staging-Production) / 메타데이터 비교 불가 |
| 3 | 평가 지표가 플랫폼 컴포넌트(`evaluate_classifier`)에 일부 묶여 있음 | 도메인 특화 지표 추가가 어색함 — 모델 개발자가 자유롭게 정의해야 함 |
| 4 | 학습이 끝나면 그걸로 끝 | 모델을 어떻게 사용할지(서빙) 자동화되어 있지 않음 |

> **핵심 원칙**: 학습 코드와 서빙 코드는 같은 feature 추출 모듈을 import 해야 한다.
> 그렇지 않으면 prod에서 미묘한 버그가 발생한다.

---

## 목표 아키텍처

```
┌────────────────────────────────────────────────────────────────────────┐
│                          mlops_platform/                               │
│                  (도메인 무관 — MLOps 팀이 관리)                         │
│                                                                        │
│  base-images/      python-cpu, pytorch-cpu                             │
│  lib/              validate_data, train_classifier, evaluate_classifier│
│                    register_model    ← NEW (Phase 2)                   │
│  serving/          KServe 배포 헬퍼  ← NEW (Phase 3)                   │
│  scripts/          공통 운영 스크립트                                    │
└────────────────────────────────────────────────────────────────────────┘
                                  ▲
                                  │ import
┌────────────────────────────────────────────────────────────────────────┐
│              projects/deepmimo_beam_selection/                         │
│                  (도메인 — 모델 개발자가 관리)                          │
│                                                                        │
│  features/         extractor.py        ← NEW (Phase 1)                 │
│                    학습/서빙 양쪽이 import                               │
│                                                                        │
│  components/       preprocess.py       (extractor 호출)                │
│                    evaluate.py         (도메인 특화 평가)               │
│                                                                        │
│  serving/          transformer.py      ← NEW (Phase 3)                 │
│                    inference_service.yaml                              │
│                                                                        │
│  pipeline.py       파이프라인 정의                                      │
│  compile.py                                                            │
└────────────────────────────────────────────────────────────────────────┘
```

### 데이터/모델 흐름

```
[ 학습 흐름 ]
  Raw Data
    └─► validate_data        (플랫폼)
          └─► preprocess     (프로젝트, features/extractor.py 호출)
                ├─► train_classifier         (플랫폼)
                │     └─► evaluate           (프로젝트)
                │           └─► register_model  (플랫폼, MLflow 등록)
                │                                       │
                                                        ▼
                                               ┌─────────────────┐
                                               │ MLflow Registry │
                                               │ Stage: Staging  │
                                               │ → Production    │
                                               └────────┬────────┘
                                                        │
                                                        ▼
[ 서빙 흐름 ]                                  ┌──────────────────┐
  HTTP Request                                 │ KServe           │
    └─► Transformer Pod ─────► Predictor Pod   │ InferenceService │
        (extractor.py 재사용)   (모델 추론)     └──────────────────┘
                                       │
                                       ▼
                                  HTTP Response (예측 빔 인덱스)
```

---

## 현재 상태와의 격차

| 구성 요소 | 현재 | 목표 | 단계 |
|-----------|------|------|------|
| Feature 추출 | `preprocess.py`에 직접 구현 | `features/extractor.py` 모듈 분리 | Phase 1 |
| 모델 저장 | KFP Output[Model] → PVC | MLflow Registry (옵션) | Phase 2 |
| 평가 컴포넌트 | 플랫폼 `evaluate_classifier` + 프로젝트 `evaluate_se` | 프로젝트 `evaluate.py`로 통합·재정리 | Phase 2 함께 |
| 서빙 | 없음 | KServe InferenceService + Transformer | Phase 3 |
| Feature Store | 없음 | (장기) Feast 또는 PVC 기반 버전 관리 | Phase 4 (선택) |

---

## 3단계 로드맵 요약

### Phase 1 — Feature Extractor 분리 (즉시 시작)

> Training-Serving Skew의 근본 원인을 제거. 가장 작고, 가장 영향이 큰 변화.

- `projects/deepmimo_beam_selection/features/extractor.py` 신규 작성
- 기존 `preprocess.py` → `extractor.py` 호출 형태로 리팩토링
- 단위 테스트 추가 (numpy 입출력만 검증)
- KFP/Docker 변경 없음

자세한 내용: [`01-phase1-feature-extractor.md`](./01-phase1-feature-extractor.md)

### Phase 2 — MLflow 모델 레지스트리 통합 (옵션)

> 모델 버저닝 + Stage 관리 + 학습 메트릭 추적.

- MLflow 서버 배포 (k3s 환경, 폐쇄망 대응)
- `mlops_platform/lib/mlops_lib/components/register_model.py` 신규
- `train_classifier`에 `use_mlflow: bool` 파라미터 추가 (기본 False, 하위 호환)
- 평가 컴포넌트를 `projects/.../evaluate.py`로 통합 정리 (도메인 메트릭은 여기서)
- 폐쇄망 wheel 추가: `mlflow`, `mlflow-skinny`

자세한 내용: [`02-phase2-mlflow.md`](./02-phase2-mlflow.md)

### Phase 3 — KServe 서빙 자동화

> 학습된 모델을 HTTP/gRPC 엔드포인트로 배포.

- KServe 설치 가이드 (k3s 폐쇄망)
- `projects/.../serving/transformer.py` 신규 (extractor 재사용)
- `projects/.../serving/inference_service.yaml` 신규
- MLflow에서 모델 자동 pull (`storageUri: mlflow://...`)
- 추론 테스트 스크립트 + curl 예시

자세한 내용: [`03-phase3-kserve.md`](./03-phase3-kserve.md)

### Phase 4 — Feature Store (장기, 선택)

> 학습/서빙 간 feature 공유 + 시간 기반 버전 관리.

DeepMIMO처럼 데이터가 정적이고 미리 생성되는 도메인에서는 우선순위가 낮다.
온라인/스트리밍 feature가 필요해질 때 검토.

후보:
- **Feast**: 오픈소스, 표준
- **PVC + 메타 JSON**: 폐쇄망 친화, 단순
- **MLflow Datasets**: 이미 MLflow를 쓴다면 가장 가벼움

---

## 역할 분리 원칙

| 구분 | MLOps | 모델 개발자 |
|------|-------|------------|
| **작성** | `mlops_platform/**` | `projects/<프로젝트명>/**` |
| **언어** | KFP DSL, Kubernetes, Docker, MLflow API, KServe | 순수 Python (numpy/torch) — KFP 임포트 0 |
| **주요 파일** | `train_classifier`, `register_model`, base 이미지 | `features/extractor.py`, `evaluate.py`, `serving/transformer.py` |
| **변경 빈도** | 낮음 (분기 단위) | 높음 (실험 단위) |
| **테스트** | 통합 테스트, K8s 스모크 | numpy/pytest 단위 테스트 |

> **모델 개발자는 KFP/Kubernetes를 몰라도 된다.**
> 단, "이 함수의 입력은 numpy ndarray, 출력도 numpy ndarray"라는 인터페이스만 지켜야 한다.

---

## 용어 정리

| 용어 | 의미 |
|------|------|
| **Training-Serving Skew** | 학습 시와 서빙 시 사용하는 feature 추출 코드가 달라서 발생하는 모델 성능 저하 |
| **Feature Store** | feature를 저장/버저닝/조회하는 중앙 저장소. 학습-서빙 일관성 보장 |
| **Model Registry** | 학습된 모델을 버전, stage, 메타데이터와 함께 관리하는 저장소 (예: MLflow) |
| **KServe Predictor** | 모델 추론 컨테이너 — 모델 가중치 로드 & forward |
| **KServe Transformer** | 전/후처리 컨테이너 — 요청을 모델 입력 형태로 변환, 응답을 가공 |
| **Stage (MLflow)** | 모델 라이프사이클 단계: None → Staging → Production → Archived |
| **InferenceService** | KServe의 CRD. predictor + transformer 조합을 K8s 리소스로 정의 |

---

## 진행 방식

1. 각 단계는 **별도 PR**로 진행한다 (작은 단위 변경 → 리뷰 용이)
2. 각 단계 시작 전 해당 `0X-phaseN-*.md` 문서를 다시 읽고 합의한다
3. 단계가 끝날 때마다 README를 갱신하고 통합 테스트 실행
4. Phase 4 (Feature Store)는 실제 필요가 발생할 때 재논의

---

## 참고 자료 (외부)

> 폐쇄망이라 직접 링크 접근이 어려울 수 있어, 실제 자료는 별도 USB로 전달.

- KServe 공식 문서 — Transformer 패턴
- MLflow Model Registry — Stage Transition
- "Hidden Technical Debt in Machine Learning Systems" (Sculley et al., NeurIPS 2015) — Training-Serving Skew 개념
