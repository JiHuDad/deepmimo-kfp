# Phase 1 — Feature Extractor 분리

> Training-Serving Skew를 막기 위해 feature 추출 로직을 별도 모듈로 분리한다.
> KFP/Docker 변경 없이 끝낼 수 있는 가장 작고 안전한 단계.

## 목표

1. Feature 추출 / 레이블 생성 로직을 `projects/.../features/extractor.py`로 추출
2. `preprocess.py`는 DeepMIMO 데이터 로딩과 train/val/test 분할만 담당
3. `extractor.py`는 KFP/Kubernetes/Docker 의존성이 **0**
4. numpy 입출력 단위 테스트 작성
5. Phase 3에서 KServe Transformer가 그대로 import할 수 있도록 한다

## 비목표 (이 단계에서 하지 않는 것)

- MLflow 도입 → Phase 2
- KServe 배포 → Phase 3
- 추론 서버 코드 작성 → Phase 3
- 새로운 Docker 이미지 빌드

---

## 배경

현재 `projects/deepmimo_beam_selection/components/preprocess.py`는 다음을 한꺼번에 한다:

1. DeepMIMO `dm.load()`
2. `dataset.compute_channels(params)` 로 채널 행렬 생성
3. **feature 추출** (`real(ch)` + `imag(ch)` concat)
4. **레이블 생성** (argmax)
5. train / val / test 분할
6. `.npy` 저장

3, 4번이 학습 전용 컴포넌트 안에 묻혀 있어, 서빙 시 동일한 변환을
재구현해야 한다. 두 코드가 어긋나는 순간 모델은 학습 때와 다른 입력을 받는다.

이 단계에서는 3, 4번만 분리한다.

---

## 새 디렉토리 구조

```
projects/deepmimo_beam_selection/
├── features/                       ← NEW
│   ├── __init__.py
│   ├── extractor.py                ← feature/label 함수 모음
│   └── README.md                   ← 인터페이스 명세
├── components/
│   └── preprocess.py               ← extractor.py 호출하도록 수정
└── tests/                          ← NEW (선택)
    └── test_extractor.py
```

---

## 파일 설계

### `features/extractor.py`

```python
"""
DeepMIMO 빔 선택을 위한 feature/label 추출 모듈.

설계 원칙:
  - KFP / Kubernetes / Docker 의존성 없음 (numpy 만 사용)
  - 학습 파이프라인과 서빙 Transformer가 동일하게 import
  - 함수 시그니처 변경은 모델 재학습이 필요한 변경으로 취급
  - 변경 시 반드시 단위 테스트 추가/수정

입력 채널 텐서 형태:
  channels: np.ndarray, shape = (N_users, n_rx_ant, n_tx_ant, n_subcarriers)
  dtype:    complex64 또는 complex128
"""

from __future__ import annotations

import numpy as np

# 학습/서빙이 동일하게 사용해야 하는 상수
DEFAULT_SUBCARRIER_INDEX: int = 0
DEFAULT_RX_ANTENNA_INDEX: int = 0


def extract_features(
    channels: np.ndarray,
    *,
    subcarrier_index: int = DEFAULT_SUBCARRIER_INDEX,
    rx_antenna_index: int = DEFAULT_RX_ANTENNA_INDEX,
) -> np.ndarray:
    """
    채널 텐서에서 학습/추론용 feature 벡터를 만든다.

    현재 정책: 첫 번째 서브캐리어, 첫 번째 RX 안테나의 채널 응답을
              real/imag 로 분리하여 concat.

    Args:
        channels: shape (N, n_rx, n_tx, n_subc), complex
        subcarrier_index: 사용할 서브캐리어 인덱스 (기본 0)
        rx_antenna_index: 사용할 RX 안테나 인덱스 (기본 0)

    Returns:
        features: shape (N, 2 * n_tx), float32
                  [real(h_0), real(h_1), ..., imag(h_0), imag(h_1), ...]
    """
    if channels.ndim != 4:
        raise ValueError(
            f"channels는 4차원 텐서여야 합니다 (N, rx, tx, subc). got shape={channels.shape}"
        )

    ch = channels[:, rx_antenna_index, :, subcarrier_index]  # (N, n_tx)
    return np.concatenate([np.real(ch), np.imag(ch)], axis=1).astype(np.float32)


def extract_labels(
    channels: np.ndarray,
    *,
    subcarrier_index: int = DEFAULT_SUBCARRIER_INDEX,
    rx_antenna_index: int = DEFAULT_RX_ANTENNA_INDEX,
) -> np.ndarray:
    """
    채널 텐서에서 최적 빔 인덱스(레이블)를 계산한다.

    정책: |h_tx|^2 가 최대인 BS 안테나 인덱스 (single-antenna codebook 가정).

    Returns:
        labels: shape (N,), int64
    """
    if channels.ndim != 4:
        raise ValueError(
            f"channels는 4차원 텐서여야 합니다. got shape={channels.shape}"
        )

    ch = channels[:, rx_antenna_index, :, subcarrier_index]  # (N, n_tx)
    return np.argmax(np.abs(ch) ** 2, axis=1).astype(np.int64)


def filter_valid_users(channels: np.ndarray) -> np.ndarray:
    """
    채널 전력이 0인 사용자(경로 없음)를 제거하기 위한 boolean mask 반환.

    Returns:
        mask: shape (N,), bool — True이면 유효 사용자
    """
    if channels.ndim != 4:
        raise ValueError(
            f"channels는 4차원 텐서여야 합니다. got shape={channels.shape}"
        )

    # 첫 RX 안테나 / 첫 서브캐리어 기준 전력 합
    ch_pow = np.abs(channels[:, 0, :, 0]).sum(axis=1)
    return ch_pow > 0


# ── 메타데이터 (서빙 시 모델 입력 차원 검증용) ────────────────────
def feature_dim(n_tx_antennas: int) -> int:
    """주어진 BS 안테나 수에서 feature 차원을 반환."""
    return 2 * n_tx_antennas


def schema() -> dict:
    """
    feature 추출 정책의 스냅샷.
    학습 시 model_meta.json 에 함께 기록하여 서빙 시 검증한다.
    """
    return {
        "version": "1.0.0",
        "subcarrier_index": DEFAULT_SUBCARRIER_INDEX,
        "rx_antenna_index": DEFAULT_RX_ANTENNA_INDEX,
        "feature_format": "concat(real, imag)",
        "label_policy": "argmax(|h|^2)",
    }
```

### `features/__init__.py`

```python
from .extractor import (
    extract_features,
    extract_labels,
    filter_valid_users,
    feature_dim,
    schema,
    DEFAULT_SUBCARRIER_INDEX,
    DEFAULT_RX_ANTENNA_INDEX,
)

__all__ = [
    "extract_features",
    "extract_labels",
    "filter_valid_users",
    "feature_dim",
    "schema",
    "DEFAULT_SUBCARRIER_INDEX",
    "DEFAULT_RX_ANTENNA_INDEX",
]
```

### `features/README.md`

```markdown
# DeepMIMO Feature Extractor

이 모듈은 학습 파이프라인과 KServe 서빙 Transformer가 **동일하게** 사용한다.

## 인터페이스

| 함수 | 입력 | 출력 |
|------|------|------|
| `extract_features(channels)` | (N, rx, tx, subc) complex | (N, 2*tx) float32 |
| `extract_labels(channels)`   | (N, rx, tx, subc) complex | (N,) int64 |
| `filter_valid_users(channels)` | 동일 | (N,) bool |

## 변경 시 주의사항

1. 시그니처를 바꾸면 기존 모델은 호환되지 않는다 (재학습 필요)
2. `schema()` 의 version을 올린다
3. 학습 코드와 서빙 Transformer 모두 새 버전으로 업데이트
4. 단위 테스트 갱신
```

### `components/preprocess.py` 수정 (diff 형태)

```python
# Before
ch_first = channels[:, 0, :, 0]
labels = np.argmax(np.abs(ch_first) ** 2, axis=1).astype(np.int64)
features = np.concatenate(
    [np.real(ch_first), np.imag(ch_first)], axis=1
).astype(np.float32)

# After
from projects.deepmimo_beam_selection.features import (
    extract_features,
    extract_labels,
    filter_valid_users,
    schema as feature_schema,
)

valid_mask = filter_valid_users(channels)
channels = channels[valid_mask]

features = extract_features(channels)
labels   = extract_labels(channels)

# 모델 메타에 feature schema 기록 (Phase 3에서 서빙 검증에 사용)
output_metrics.log_metric("feature_schema_version", feature_schema()["version"])
```

> 주의: `preprocess` 컴포넌트는 KFP `@dsl.component` 데코레이터로 래핑되어
> 컨테이너 안에서 실행된다. import가 동작하려면 `projects/` 디렉토리가
> Docker 이미지의 PYTHONPATH 위에 있어야 한다.
> deepmimo-base 이미지 빌드 시 `COPY projects/ /app/projects/` 가
> 이미 있는지 확인하고, 없다면 추가한다 (다음 절 참조).

---

## Docker 이미지 변경

`projects/deepmimo_beam_selection/docker/deepmimo-base/Dockerfile` 에
다음을 추가한다 (이미 있다면 생략):

```dockerfile
# features/ 모듈을 이미지에 포함
COPY projects/deepmimo_beam_selection/features/ /app/projects/deepmimo_beam_selection/features/
COPY projects/deepmimo_beam_selection/__init__.py /app/projects/deepmimo_beam_selection/

# (이미 설정되어 있다면 생략)
ENV PYTHONPATH=/app
```

이 변경으로 `from projects.deepmimo_beam_selection.features import ...`가
컨테이너 안에서 동작한다.

---

## 단위 테스트

### `tests/test_extractor.py`

```python
"""
features/extractor.py 단위 테스트.
폐쇄망 환경에서도 실행 가능하도록 numpy 만 사용.

실행:
    cd /home/user/deepmimo-kfp
    python -m pytest projects/deepmimo_beam_selection/tests/ -v
"""

import numpy as np
import pytest

from projects.deepmimo_beam_selection.features import (
    extract_features,
    extract_labels,
    filter_valid_users,
    feature_dim,
)


@pytest.fixture
def fake_channels():
    """N=10, rx=1, tx=8, subc=4 형태의 가짜 채널."""
    rng = np.random.default_rng(42)
    return rng.normal(size=(10, 1, 8, 4)) + 1j * rng.normal(size=(10, 1, 8, 4))


def test_extract_features_shape(fake_channels):
    feats = extract_features(fake_channels)
    assert feats.shape == (10, 16)            # 2 * n_tx
    assert feats.dtype == np.float32


def test_extract_features_real_imag_split(fake_channels):
    feats = extract_features(fake_channels)
    n_tx = fake_channels.shape[2]
    real_part = feats[:, :n_tx]
    imag_part = feats[:, n_tx:]
    expected_real = np.real(fake_channels[:, 0, :, 0])
    expected_imag = np.imag(fake_channels[:, 0, :, 0])
    np.testing.assert_allclose(real_part, expected_real, rtol=1e-5)
    np.testing.assert_allclose(imag_part, expected_imag, rtol=1e-5)


def test_extract_labels_shape(fake_channels):
    labels = extract_labels(fake_channels)
    assert labels.shape == (10,)
    assert labels.dtype == np.int64
    assert labels.min() >= 0
    assert labels.max() < 8


def test_extract_labels_correctness():
    # 각 사용자마다 정답 빔이 명확하도록 수동 구성
    ch = np.zeros((3, 1, 4, 1), dtype=np.complex64)
    ch[0, 0, 1, 0] = 10 + 0j   # 0번 사용자 → 빔 1
    ch[1, 0, 3, 0] = 5 + 5j    # 1번 사용자 → 빔 3
    ch[2, 0, 0, 0] = 1 + 0j    # 2번 사용자 → 빔 0
    labels = extract_labels(ch)
    np.testing.assert_array_equal(labels, [1, 3, 0])


def test_filter_valid_users():
    ch = np.zeros((4, 1, 4, 2), dtype=np.complex64)
    ch[0, 0, 0, 0] = 1.0       # 유효
    ch[2, 0, 2, 0] = 0.5j      # 유효
    # 1, 3번 사용자는 0
    mask = filter_valid_users(ch)
    np.testing.assert_array_equal(mask, [True, False, True, False])


def test_feature_dim():
    assert feature_dim(8) == 16
    assert feature_dim(64) == 128


def test_invalid_input_dimension():
    bad = np.zeros((5, 8))  # 2차원
    with pytest.raises(ValueError, match="4차원"):
        extract_features(bad)
    with pytest.raises(ValueError, match="4차원"):
        extract_labels(bad)
```

---

## 작업 순서

1. `projects/deepmimo_beam_selection/features/` 디렉토리 생성
2. `extractor.py`, `__init__.py`, `README.md` 작성
3. `components/preprocess.py` 리팩토링 (extractor import)
4. (선택) `tests/test_extractor.py` 작성 + pytest 실행
5. `deepmimo-base` Dockerfile에 `COPY projects/.../features/` 추가
6. 이미지 재빌드: `make build` 또는 해당 스크립트
7. 파이프라인 1회 실행 후 결과 검증
   - feature shape, label 분포가 이전과 동일한지 확인
   - `output_metrics.feature_schema_version == "1.0.0"` 확인
8. PR 작성, 리뷰, 머지

---

## 검증 체크리스트

- [ ] `extractor.py`가 KFP/Kubernetes/torch를 import하지 않는다 (`grep` 으로 확인)
- [ ] `pytest` 통과
- [ ] 리팩토링 전후 `features.npy`, `labels.npy`의 shape이 동일
- [ ] 리팩토링 전후 학습된 모델의 `best_val_accuracy`가 동일 (시드 고정)
- [ ] `model_meta.json`에 `feature_schema_version` 기록
- [ ] Phase 3에서 import할 수 있도록 `__init__.py`에 export

---

## 위험 요소

| 위험 | 완화 방안 |
|------|-----------|
| 리팩토링 전후 미묘한 수치 차이로 정확도 변화 | 시드 고정 후 best_val_accuracy 비교 |
| `projects/` 가 컨테이너 PYTHONPATH에 없음 | Dockerfile에 명시적으로 COPY + PYTHONPATH 설정 |
| import 경로 충돌 (`projects.deepmimo_beam_selection`) | 현재 main 브랜치 구조와 동일하게 유지 |

---

## 작업량 추정

- 코드 작성: 30분
- 테스트 작성: 30분
- Docker 이미지 재빌드 + 검증: 30~60분
- 총: **반나절 이내** 완료 가능

---

## 다음 단계

Phase 1이 머지되면 `02-phase2-mlflow.md` 검토 → Phase 2 진행.
