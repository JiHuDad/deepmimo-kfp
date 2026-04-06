# deepmimo-kfp

DeepMIMO 레이트레이싱 채널 데이터를 이용한 빔 선택 모델 학습 파이프라인.
폐쇄망(Air-gapped) Ubuntu 24.04 + k3s + Kubeflow Pipelines v2 환경 기준.

## 파이프라인 개요

```
[load_scenario]  ← PVC (O1_60 시나리오, 사전 적재)
       ↓
[preprocess]     — DeepMIMO v4 load() + compute_channels() → features.npy
       ↓
[train]          — MLP Beam Selection (PyTorch) → best_model.pt
       ↓
[evaluate]       — Top-1/3 정확도, Spectral Efficiency → KFP UI 메트릭
```

## 최초 1회 수동 설정 (sudo 필요)

> `make` 명령들은 sudo 없이 실행되지만, 아래 항목들은 최초 1회 직접 실행 필요.

### 1. 시나리오 데이터 디렉토리 생성

```bash
sudo mkdir -p /home/fall/data/deepmimo-scenarios
sudo chown $USER /home/fall/data/deepmimo-scenarios
```

### 2. k3s가 localhost:5000을 신뢰하도록 확인

```bash
# 이미 설정되어 있어야 함 (install_kubeflow 과정에서)
cat /etc/rancher/k3s/registries.yaml
# configs."localhost:5000".tls.insecure_skip_verify: true 확인
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

### 3. DeepMIMO O1_60 시나리오 데이터 배치

deepmimo.net에서 O1_60 시나리오를 다운로드하여:
```
/home/fall/data/deepmimo-scenarios/O1_60/
```
에 압축 해제.

---

## 빠른 시작

### 전제조건

- k3s 및 Kubeflow Pipelines v2 설치 완료
- `localhost:5000` 로컬 Docker 레지스트리 실행 중
- 인터넷이 되는 별도 머신에서 오프라인 패키지 수집 완료

### Step 1: 오프라인 패키지 수집 (온라인 머신에서)

```bash
bash offline-packages/collect.sh
# 결과물을 USB로 이 서버에 복사
```

### Step 2: 환경 설정 (폐쇄망 서버에서)

```bash
# KFP SDK 설치
make install-sdk

# Docker 이미지 빌드 및 push
make build

# PVC 생성 및 시나리오 데이터 적재
# (먼저 /home/fall/data/deepmimo-scenarios/O1_60/ 에 시나리오 파일 배치)
make setup
```

### Step 3: 파이프라인 실행

```bash
make run
# KFP UI: http://192.168.1.112:31380
```

또는 전체 한 번에:

```bash
make all
```

## 디렉토리 구조

```
deepmimo-kfp/
├── docker/              # Docker 이미지 정의
│   ├── base/            # DeepMIMO + numpy/scipy
│   └── trainer/         # base + PyTorch
├── components/          # KFP v2 컴포넌트
│   ├── load_scenario/   # PVC에서 시나리오 로드
│   ├── preprocess/      # DeepMIMO 채널 생성
│   ├── train/           # MLP 학습
│   └── evaluate/        # 성능 평가
├── pipelines/           # 파이프라인 정의 및 컴파일
├── k8s/                 # PVC, Job 매니페스트
├── scripts/             # 운영 자동화 스크립트
└── offline-packages/    # 폐쇄망 패키지 수집 스크립트
```

## 환경 변수

| 변수 | 기본값 | 설명 |
|------|--------|------|
| `IMAGE_TAG` | `latest` | Docker 이미지 태그 |
| `KFP_ENDPOINT` | `http://192.168.1.112:31380` | KFP API 엔드포인트 |
| `PIPELINE_NAME` | `deepmimo-beam-selection` | KFP 파이프라인 이름 |
| `EXPERIMENT_NAME` | `deepmimo-experiments` | KFP 실험 이름 |

## 파이프라인 파라미터

| 파라미터 | 기본값 | 설명 |
|----------|--------|------|
| `scenario_name` | `O1_60` | DeepMIMO 시나리오 이름 |
| `num_epochs` | `50` | 학습 에폭 수 |
| `learning_rate` | `0.001` | 학습률 |
| `batch_size` | `256` | 배치 크기 |
| `hidden_dims` | `256,128,64` | MLP 히든 레이어 차원 |
| `train_ratio` | `0.7` | 학습 데이터 비율 |
| `val_ratio` | `0.15` | 검증 데이터 비율 |
