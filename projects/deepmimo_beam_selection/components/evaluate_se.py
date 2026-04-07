"""
evaluate_se 컴포넌트 (DeepMIMO 전용)

DFT 코드북 기반 빔포밍 Spectral Efficiency(SE) 비율을 계산한다.
범용 evaluate_classifier의 예측 결과와 테스트 채널 데이터를 입력받는다.

SE 비율 = SE(predicted beam) / SE(optimal beam)
"""

import os

from kfp import dsl
from kfp.dsl import Input, Output, Dataset, Metrics

_IMAGE_TAG = os.environ.get("IMAGE_TAG", "latest")


@dsl.component(
    base_image=f"localhost:5000/python-cpu:{_IMAGE_TAG}",
    packages_to_install=[],
)
def evaluate_se(
    test_dataset: Input[Dataset],
    predictions: Input[Dataset],
    output_metrics: Output[Metrics],
) -> None:
    """
    DFT 코드북 기반 빔포밍 Spectral Efficiency 비율 계산.

    test_dataset: channel.npy, labels.npy 포함
    predictions:  top1_preds.npy 포함 (evaluate_classifier 출력)
    """
    import numpy as np

    # ── 데이터 로드 ─────────────────────────────────────────
    channel_test = np.load(os.path.join(test_dataset.path, "channel.npy"))
    y_test       = np.load(os.path.join(predictions.path, "labels.npy"))
    top1_preds   = np.load(os.path.join(predictions.path, "top1_preds.npy"))

    # ── Spectral Efficiency 비율 ─────────────────────────────
    # SE ∝ log2(1 + SNR * |h^H * f|^2) 에서 SNR=20dB(100)
    ch = channel_test[:, 0, :, 0]  # (N, n_tx_ant) — 첫 서브캐리어, 첫 RX 안테나
    print(f"[evaluate_se] ch shape: {ch.shape}, dtype: {ch.dtype}")
    print(f"[evaluate_se] ch mean abs: {np.abs(ch).mean():.2e}, max: {np.abs(ch).max():.2e}")

    # 채널을 사용자별 RMS로 정규화 (채널 크기의 절대값에 무관하게 SE 비율 계산)
    ch_rms = np.sqrt((np.abs(ch) ** 2).mean(axis=1, keepdims=True)) + 1e-30
    ch_norm = ch / ch_rms  # (N, n_ant), 정규화된 채널

    def beam_se(h_vec, beam_indices):
        """DFT 코드북 기반 빔포밍 SE 계산 (SNR=20dB 가정)."""
        n_ant = h_vec.shape[1]
        snr = 100.0  # 20dB
        beams = np.exp(
            2j * np.pi * np.arange(n_ant)[:, None]
            * np.arange(n_ant)[None, :] / n_ant
        ) / np.sqrt(n_ant)  # (n_ant, n_ant) DFT codebook
        selected_beams = beams[:, beam_indices]  # (n_ant, N)
        bf_gain = np.abs(np.einsum("ni,in->n", h_vec, selected_beams)) ** 2  # (N,)
        return np.log2(1 + snr * bf_gain)

    se_optimal   = beam_se(ch_norm, y_test)
    se_predicted = beam_se(ch_norm, top1_preds)
    se_ratio = np.mean(se_predicted / (se_optimal + 1e-10))
    print(f"[evaluate_se] SE Ratio: {se_ratio:.4f}")

    # ── KFP 메트릭 ───────────────────────────────────────────
    output_metrics.log_metric("se_ratio", round(float(se_ratio), 4))
    output_metrics.log_metric("test_samples", len(y_test))
