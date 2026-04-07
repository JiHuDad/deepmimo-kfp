"""
DeepMIMO 빔 선택을 위한 feature / label 추출 모듈.

설계 원칙
----------
- KFP / Kubernetes / Docker / PyTorch 의존성 없음. numpy 만 사용.
- 학습 파이프라인(preprocess.py)과 서빙 Transformer(Phase 3)가
  동일하게 import 하여 Training-Serving Skew를 방지한다.
- 함수 시그니처 · 정책 변경은 모델 재학습이 필요한 변경으로 취급한다.
  변경 시 schema()의 version을 올리고 단위 테스트를 갱신한다.

입력 채널 텐서 형태
-------------------
  channels : np.ndarray
    shape = (N_users, n_rx_ant, n_tx_ant, n_subcarriers)
    dtype = complex64 또는 complex128
"""

from __future__ import annotations

import numpy as np

# ── 정책 상수 ─────────────────────────────────────────────────────────
# 학습과 서빙이 반드시 동일한 값을 사용해야 한다.
DEFAULT_SUBCARRIER_INDEX: int = 0
DEFAULT_RX_ANTENNA_INDEX: int = 0


# ── 공개 API ──────────────────────────────────────────────────────────

def extract_features(
    channels: np.ndarray,
    *,
    subcarrier_index: int = DEFAULT_SUBCARRIER_INDEX,
    rx_antenna_index: int = DEFAULT_RX_ANTENNA_INDEX,
) -> np.ndarray:
    """채널 텐서 → feature 벡터 (학습·추론 공통 입력).

    정책: 지정 서브캐리어 × RX 안테나의 채널 응답을 real / imag 분리 후 concat.

    Parameters
    ----------
    channels : shape (N, n_rx, n_tx, n_subc), complex
    subcarrier_index : 사용할 서브캐리어 인덱스 (기본 0)
    rx_antenna_index : 사용할 RX 안테나 인덱스 (기본 0)

    Returns
    -------
    features : shape (N, 2 * n_tx), float32
        [real(h_0), ..., real(h_{n_tx-1}), imag(h_0), ..., imag(h_{n_tx-1})]
    """
    _check_4d(channels)
    ch = channels[:, rx_antenna_index, :, subcarrier_index]   # (N, n_tx)
    return np.concatenate(
        [np.real(ch), np.imag(ch)], axis=1
    ).astype(np.float32)


def extract_labels(
    channels: np.ndarray,
    *,
    subcarrier_index: int = DEFAULT_SUBCARRIER_INDEX,
    rx_antenna_index: int = DEFAULT_RX_ANTENNA_INDEX,
) -> np.ndarray:
    """채널 텐서 → 최적 빔 인덱스 레이블 (학습 전용).

    정책: |h_tx|^2 가 최대인 BS 안테나 인덱스.

    Returns
    -------
    labels : shape (N,), int64
    """
    _check_4d(channels)
    ch = channels[:, rx_antenna_index, :, subcarrier_index]   # (N, n_tx)
    return np.argmax(np.abs(ch) ** 2, axis=1).astype(np.int64)


def filter_valid_users(channels: np.ndarray) -> np.ndarray:
    """채널 전력이 0인 사용자(경로 없음) 제거용 boolean mask 반환.

    Returns
    -------
    mask : shape (N,), bool  — True 이면 유효 사용자
    """
    _check_4d(channels)
    ch_pow = np.abs(channels[:, DEFAULT_RX_ANTENNA_INDEX, :, DEFAULT_SUBCARRIER_INDEX]).sum(axis=1)
    return ch_pow > 0


def feature_dim(n_tx_antennas: int) -> int:
    """BS 안테나 수 → feature 벡터 차원."""
    return 2 * n_tx_antennas


def schema() -> dict:
    """feature 추출 정책의 스냅샷.

    학습 시 model_meta.json 에 함께 기록하여 서빙 시 호환성 검증에 사용.
    버전을 올릴 때는 기존 모델과의 호환성을 반드시 확인할 것.
    """
    return {
        "version": "1.0.0",
        "subcarrier_index": DEFAULT_SUBCARRIER_INDEX,
        "rx_antenna_index": DEFAULT_RX_ANTENNA_INDEX,
        "feature_format": "concat(real, imag)",
        "label_policy": "argmax(|h|^2)",
    }


# ── 내부 헬퍼 ─────────────────────────────────────────────────────────

def _check_4d(channels: np.ndarray) -> None:
    if channels.ndim != 4:
        raise ValueError(
            f"channels 는 4차원 텐서여야 합니다 (N, rx, tx, subc). "
            f"실제 shape={channels.shape}"
        )
