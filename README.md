# deepmimo-kfp

DeepMIMO 레이트레이싱 채널 데이터를 이용한 빔 선택 모델 학습 파이프라인.
폐쇄망(Air-gapped) Ubuntu 24.04 + k3s + Kubeflow Pipelines v2 환경 기준.

## 파이프라인 개요

```
[load_scenario]  ← deepmimo-scenarios PVC (hostPath, 복사 없음)
       ↓            경로 검증 후 절대 경로를 아티팩트로 전달
[preprocess]     — DeepMIMO v4 dm.load() + compute_channels() → features/labels/channel.npy
       ↓
[train]          — MLP Beam Selection (PyTorch) → best_model.pt
       ↓
[evaluate]       — Top-1/3 정확도, Spectral Efficiency → KFP UI 메트릭
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
make build             # Docker 이미지 빌드 및 push
make setup             # hostPath PV/PVC 생성 (데이터 복사 없음)
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

## 디렉토리 구조

```
deepmimo-kfp/
├── docker/              # Docker 이미지 정의
│   ├── base/            # DeepMIMO + numpy/scipy
│   └── trainer/         # base + PyTorch
├── components/          # KFP v2 컴포넌트
│   ├── load_scenario/   # PVC 경로 검증 및 아티팩트 전달
│   ├── preprocess/      # DeepMIMO 채널 생성 및 분할
│   ├── train/           # MLP 학습
│   └── evaluate/        # 성능 평가
├── pipelines/           # 파이프라인 정의 및 컴파일
├── k8s/                 # PV/PVC 매니페스트
│   ├── pv-scenarios.yaml    # hostPath PV + PVC (시나리오, 읽기전용)
│   └── pvc-artifacts.yaml   # 학습 아티팩트 PVC
├── scripts/             # 운영 자동화 스크립트
└── offline-packages/    # 폐쇄망 패키지 수집 스크립트
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

## 알려진 제약

- **tx/rx set 인덱스**: O1_60 시나리오에서 BS는 TX set 3, UE는 RX set 0이 일반적입니다.
  다른 시나리오를 사용할 경우 `tx_set_id`/`rx_set_id`를 확인 후 조정하세요.
- **KFP UI NodePort**: 환경에 따라 NodePort(31380)가 비활성화되어 있을 수 있습니다.
  `kubectl port-forward`를 통해 접근하세요.
