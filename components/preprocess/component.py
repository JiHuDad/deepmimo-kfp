"""
preprocess 컴포넌트 (DeepMIMO v4 API)

DeepMIMO v4 워크플로우:
  dm.load(scenario, dataset_folder=...) → dataset
  dm.ChannelParameters() → params
  dataset.compute_channels(params) → channels  shape: [N, n_rx, n_tx, n_subcarr]

채널 데이터를 생성하고 train/val/test 로 분리한다.
"""

from kfp import dsl
from kfp.dsl import Input, Output, Dataset, Metrics


@dsl.component(
    base_image="localhost:5000/deepmimo-base:latest",
    packages_to_install=[],
)
def preprocess(
    scenario_dataset: Input[Dataset],
    scenario_name: str,
    bs_antenna_shape: str,
    num_subcarriers: int,
    bandwidth: float,
    num_paths: int,
    train_ratio: float,
    val_ratio: float,
    output_train: Output[Dataset],
    output_val: Output[Dataset],
    output_test: Output[Dataset],
    output_metrics: Output[Metrics],
) -> None:
    """
    DeepMIMO v4 채널 생성 및 train/val/test 분리.

    bs_antenna_shape: 쉼표 구분 문자열, 예: "8,1" → [8, 1]
    num_paths: 0 이면 전체 경로 사용 (None)
    """
    import os

    import deepmimo as dm
    import numpy as np

    # ── DeepMIMO 데이터셋 로드 ──────────────────────────────
    dataset_folder = scenario_dataset.path
    print(f"[preprocess] 시나리오 로드: {scenario_name} ← {dataset_folder}")
    print(f"[preprocess] 폴더 내용: {os.listdir(dataset_folder)}")

    dataset = dm.load(scenario_name, dataset_folder=dataset_folder)
    print(f"[preprocess] 로드 완료. 사용자 수: {dataset.num_users}")
    dataset.info()

    # ── 채널 파라미터 설정 (ChannelParameters) ──────────────
    params = dm.ChannelParameters()

    # BS 안테나 (예: "8,1" → shape=[8,1])
    ant_shape = [int(x) for x in bs_antenna_shape.split(",")]
    params.bs_antenna.shape = ant_shape
    params.bs_antenna.spacing = 0.5

    # UE 안테나 (단일 안테나 수신기)
    params.ue_antenna.shape = [1, 1]
    params.ue_antenna.spacing = 0.5

    # OFDM 파라미터
    params.num_subcarriers = num_subcarriers
    params.bandwidth = bandwidth

    # 경로 수 제한 (0이면 전체)
    params.num_paths = num_paths if num_paths > 0 else None

    # ── 채널 생성 ────────────────────────────────────────────
    print(f"[preprocess] 채널 생성 중... (안테나: {ant_shape}, 서브캐리어: {num_subcarriers})")
    channels = dataset.compute_channels(params)
    # shape: [N_users, n_rx_ant, n_tx_ant, n_subcarriers]
    print(f"[preprocess] 채널 shape: {channels.shape}, dtype: {channels.dtype}")

    n_users = channels.shape[0]
    n_tx = channels.shape[2]  # BS 안테나 수

    # ── 빔 선택 레이블 생성 ───────────────────────────────────
    # 첫 번째 서브캐리어 기준, UE 안테나 0번 기준으로 최적 BS 빔 인덱스 결정
    ch_first = channels[:, 0, :, 0]  # (N_users, n_tx) — 첫 서브캐리어
    labels = np.argmax(np.abs(ch_first) ** 2, axis=1).astype(np.int64)  # (N_users,)

    # ── 특징 벡터: 첫 서브캐리어의 real/imag concat ──────────
    features = np.concatenate(
        [np.real(ch_first), np.imag(ch_first)], axis=1
    ).astype(np.float32)  # (N_users, n_tx * 2)

    print(f"[preprocess] features: {features.shape}, labels: {labels.shape}, classes: {labels.max()+1}")

    # ── train / val / test 분리 ──────────────────────────────
    idx = np.random.permutation(n_users)
    n_train = int(n_users * train_ratio)
    n_val   = int(n_users * val_ratio)

    def _save(output: Output[Dataset], indices, name: str):
        os.makedirs(output.path, exist_ok=True)
        np.save(os.path.join(output.path, "features.npy"), features[indices])
        np.save(os.path.join(output.path, "labels.npy"),   labels[indices])
        # 전체 채널도 저장 (evaluate에서 SE 계산 용도)
        np.save(os.path.join(output.path, "channel.npy"),  channels[indices])
        print(f"[preprocess] {name}: {len(indices)}개 샘플 → {output.path}")

    _save(output_train, idx[:n_train],            "train")
    _save(output_val,   idx[n_train:n_train+n_val], "val")
    _save(output_test,  idx[n_train+n_val:],       "test")

    # ── KFP 메트릭 ───────────────────────────────────────────
    output_metrics.log_metric("total_users",   n_users)
    output_metrics.log_metric("n_train",       n_train)
    output_metrics.log_metric("n_val",         n_val)
    output_metrics.log_metric("n_test",        n_users - n_train - n_val)
    output_metrics.log_metric("n_bs_antennas", n_tx)
    output_metrics.log_metric("n_subcarriers", num_subcarriers)
    output_metrics.log_metric("feature_dim",   features.shape[1])
    output_metrics.log_metric("num_classes",   int(labels.max()) + 1)
