# MLOps KFP Platform

폐쇄망(Air-gapped) Ubuntu 24.04 + k3s + Kubeflow Pipelines v2 기반 MLOps 플랫폼.
범용 컴포넌트(데이터 검증, 학습, 평가)를 제공하며, 프로젝트별 도메인 로직은 분리하여 관리한다.

## 디렉토리 구조

```
deepmimo-kfp/                        ← 저장소 루트
├── mlops_platform/                    ← MLOps 플랫폼 (범용 인프라)
│   ├── lib/mlops_lib/                ← 범용 Python 라이브러리
│   │   ├── components/               ← KFP v2 범용 컴포넌트
│   │   │   ├── validate_data.py      │  PVC 경로 검증
│   │   │   ├── train_classifier.py   │  MLP 분류 모델 학습
│   │   │   └── evaluate_classifier.py│  Top-1/3 정확도, 학습 곡선
│   │   └── pipeline_helpers.py       ← PVC 마운트 등 유틸리티
│   ├── base-images/                  ← Docker 베이스 이미지
│   │   ├── python-cpu/               │  numpy/scipy/matplotlib
│   │   └── pytorch-cpu/              │  + PyTorch CPU
│   ├── k8s/                          ← K8s 매니페스트 템플릿
│   │   ├── pv-data.yaml              │  hostPath PV/PVC (범용)
│   │   └── pvc-artifacts.yaml        │  학습 아티팩트 PVC
│   └── scripts/                      ← 플랫폼 운영 스크립트
│       ├── lib/common.sh             │  공통 변수/함수
│       ├── build-base-images.sh      │  python-cpu, pytorch-cpu 빌드
│       ├── install-kfp-sdk.sh        │  KFP SDK 오프라인 설치
│       ├── copy-scenarios.sh         │  시나리오 복사
│       └── setup-k8s.sh              │  플랫폼 K8s 리소스 생성
├── projects/                         ← 프로젝트별 도메인 로직
│   └── deepmimo_beam_selection/      ← DeepMIMO 빔 선택 프로젝트
│       ├── components/               ← 프로젝트 전용 컴포넌트
│       │   ├── preprocess.py         │  DeepMIMO v4 채널 생성
│       │   └── evaluate_se.py        │  Spectral Efficiency 평가
│       ├── pipeline.py               ← 파이프라인 정의
│       ├── compile.py                ← YAML 컴파일
│       ├── docker/deepmimo/          ← 프로젝트 전용 이미지
│       └── scripts/                  ← 프로젝트 빌드/실행 스크립트
├── offline-packages/                 ← 폐쇄망 패키지 수집
│   └── collect.sh
├── Makefile                          ← 통합 빌드/실행 진입점
└── README.md
```

## 플랫폼 vs 프로젝트 분리

| 구분 | 플랫폼 (`mlops_platform/`) | 프로젝트 (`projects/`) |
|------|----------------------|----------------------|
| 역할 | 범용 MLOps 인프라 | 도메인 특화 로직 |
| 컴포넌트 | validate_data, train_classifier, evaluate_classifier | preprocess, evaluate_se |
| Docker 이미지 | python-cpu, pytorch-cpu | deepmimo (DeepMIMO 포함) |
| K8s 리소스 | artifacts PVC, PV 템플릿 | 시나리오 PV/PVC |
| 재사용 | 모든 프로젝트 공통 | 프로젝트별 고유 |

## 파이프라인 개요 (DeepMIMO 빔 선택)

```
[validate_data]  ← 플랫폼 범용: PVC 경로 검증
       ↓
[preprocess]     ← 프로젝트 전용: DeepMIMO v4 채널 생성 → features/labels
       ↓
[train_classifier] ← 플랫폼 범용: MLP 학습 → best_model.pt
       ↓
[evaluate_classifier] ← 플랫폼 범용: Top-1/3 정확도, 학습 곡선
       ↓
[evaluate_se]    ← 프로젝트 전용: DFT 빔포밍 SE 비율
```

시나리오 데이터는 **복사하지 않습니다.**
`deepmimo-scenarios` PVC는 호스트 경로를 hostPath PV로 직접 참조하므로,
파이프라인 실행 중 추가 디스크 I/O가 발생하지 않습니다.

## 최초 1회 수동 설정 (sudo 필요)

### 1. 시나리오 데이터 디렉토리 생성

```bash
mkdir -p ~/data/deepmimo-scenarios
```

### 2. k3s가 localhost:5000을 신뢰하도록 확인

```bash
cat /etc/rancher/k3s/registries.yaml
# mirrors."localhost:5000" 항목 확인
```

설정이 없다면:
```bash
sudo tee /etc/rancher/k3s/registries.yaml <<'EOF'
mirrors:
  "localhost:5000":
    endpoint:
      - "http://localhost:5000"
configs:
  "localhost:5000":
    tls:
      insecure_skip_verify: true
EOF
sudo systemctl restart k3s
```

### 3. DeepMIMO 시나리오 데이터 배치

deepmimo.net에서 O1_60 시나리오를 다운로드하거나 `make collect`로 자동 수집 후:

```
~/data/deepmimo-scenarios/O1_60/
```

에 압축 해제. 경로를 바꾸려면 `SCENARIO_HOST_PATH` 환경변수로 지정:

```bash
export SCENARIO_HOST_PATH=/your/custom/path
make setup
```

---

## 빠른 시작

### 전제조건

- k3s 및 Kubeflow Pipelines v2 설치 완료
- `localhost:5000` 로컬 Docker 레지스트리 실행 중
- `~/data/deepmimo-scenarios/O1_60/` 에 시나리오 데이터 존재

### Step 1: 패키지 및 시나리오 수집 (인터넷 되는 머신에서)

```bash
make collect
# pip 패키지(whl), Docker 이미지, DeepMIMO 시나리오를 자동으로 수집
# → offline-packages/wheels/
# → offline-packages/python-3.12-slim.tar
# → offline-packages/scenarios/O1_60/
```

수집 완료 후 `offline-packages/` 전체를 USB로 폐쇄망 서버에 복사.

### Step 2: 환경 설정 (폐쇄망 서버에서)

```bash
make install-sdk       # 가상환경 생성 + KFP SDK 설치
make copy-scenarios    # USB의 시나리오 → ~/data/deepmimo-scenarios/
make build             # Docker 이미지 빌드 및 push (플랫폼 + 프로젝트)
make setup             # K8s PV/PVC 생성 (데이터 복사 없음)
```

### Step 3: 파이프라인 실행

```bash
make run
```

또는 전체 한 번에:

```bash
make all
```

KFP UI는 `kubectl port-forward`로 접근:

```bash
kubectl port-forward -n kubeflow svc/ml-pipeline-ui 8080:80
# 브라우저에서 http://localhost:8080
```

---

## 환경 변수

| 변수 | 기본값 | 설명 |
|------|--------|------|
| `IMAGE_TAG` | `latest` | Docker 이미지 태그 (컴파일 시 컴포넌트에 반영) |
| `SCENARIO_HOST_PATH` | `~/data/deepmimo-scenarios` | 시나리오 데이터 호스트 경로 |
| `PIPELINE_NAME` | `deepmimo-beam-selection` | KFP 파이프라인 이름 |
| `EXPERIMENT_NAME` | `deepmimo-experiments` | KFP 실험 이름 |

`IMAGE_TAG`를 바꾸려면 컴파일 전에 설정하세요:

```bash
export IMAGE_TAG=v1.0
make run
```

---

## 파이프라인 파라미터

| 파라미터 | 기본값 | 설명 |
|----------|--------|------|
| `scenario_name` | `O1_60` | DeepMIMO 시나리오 이름 |
| `scenario_source_path` | `/data/scenarios` | PVC 내 시나리오 루트 경로 |
| `bs_antenna_shape` | `8,1` | BS 안테나 배열 형태 (n_h,n_v) |
| `num_subcarriers` | `512` | OFDM 서브캐리어 수 |
| `bandwidth` | `50.0` | 대역폭 (MHz) |
| `num_paths` | `5` | 로드할 최대 경로 수 (0 = DeepMIMO 기본값 10) |
| `tx_set_id` | `3` | BS TX set 인덱스 |
| `rx_set_id` | `0` | UE RX set 인덱스 |
| `max_users` | `50000` | 사용할 최대 UE 수 (0 = 전체, 메모리 주의) |
| `train_ratio` | `0.7` | 학습 데이터 비율 |
| `val_ratio` | `0.15` | 검증 데이터 비율 |
| `random_seed` | `42` | 데이터 분할 재현성 시드 |
| `num_epochs` | `50` | 학습 에폭 수 |
| `learning_rate` | `0.001` | 학습률 |
| `batch_size` | `256` | 배치 크기 |
| `hidden_dims` | `256,128,64` | MLP 히든 레이어 차원 (쉼표 구분) |

---

## 새 프로젝트 추가 방법

1. `projects/my_new_project/` 디렉토리 생성
2. 도메인 전용 컴포넌트를 `components/`에 작성
3. `pipeline.py`에서 플랫폼 컴포넌트 + 프로젝트 컴포넌트를 조합
4. 필요 시 프로젝트 전용 Docker 이미지를 `docker/`에 정의
5. `scripts/`에 빌드/실행 스크립트 작성
6. 루트 Makefile에 프로젝트 타겟 추가

```python
# pipeline.py 예시
from mlops_platform.lib.mlops_lib.components import validate_data, train_classifier, evaluate_classifier
from projects.my_new_project.components import my_preprocess
```

---

## 알려진 제약

- **tx/rx set 인덱스**: O1_60 시나리오에서 BS는 TX set 3, UE는 RX set 0이 일반적입니다.
  다른 시나리오를 사용할 경우 `tx_set_id`/`rx_set_id`를 확인 후 조정하세요.
- **KFP UI NodePort**: 환경에 따라 NodePort(31380)가 비활성화되어 있을 수 있습니다.
  `kubectl port-forward`를 통해 접근하세요.
- **디렉토리 이름**: `mlops_platform` 디렉토리는 Python `platform` 표준 모듈과의
  충돌을 피하기 위해 접두사 `mlops_`를 사용합니다.
