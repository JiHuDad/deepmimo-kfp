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
    tx_set_id: int,
    rx_set_id: int,
    max_users: int,
    output_train: Output[Dataset],
    output_val: Output[Dataset],
    output_test: Output[Dataset],
    output_metrics: Output[Metrics],
) -> None:
    """
    DeepMIMO v4 채널 생성 및 train/val/test 분리.

    bs_antenna_shape: 쉼표 구분 문자열, 예: "8,1" → [8, 1]
    num_paths: 0 이면 전체 경로 사용 (None)
    tx_set_id: 사용할 TX set 인덱스 (BS)
    rx_set_id: 사용할 RX set 인덱스 (UE)
    max_users: 0 이면 전체 사용자 (메모리 이슈 시 제한)
    """
    import os

    import deepmimo as dm
    import numpy as np

    # ── DeepMIMO 데이터셋 로드 ──────────────────────────────
    # load_scenario가 기록한 PVC 내 절대 경로를 읽음 (데이터 복사 없음)
    with open(scenario_dataset.path) as f:
        scen_abs_path = f.read().strip()
    print(f"[preprocess] 시나리오 경로: {scen_abs_path}")
    print(f"[preprocess] 폴더 내용: {os.listdir(scen_abs_path)[:10]}")

    # dm.load()는 내부에서 scen_name.lower()를 수행하므로
    # Linux 대소문자 구분 파일시스템에서 O1_60 ≠ o1_60 문제 발생.
    # /tmp에 소문자 심볼릭 링크를 생성하여 우회한다.
    scenarios_tmp = "/tmp/dm_scenarios"
    os.makedirs(scenarios_tmp, exist_ok=True)
    symlink_path = os.path.join(scenarios_tmp, scenario_name.lower())
    if not os.path.exists(symlink_path):
        os.symlink(scen_abs_path, symlink_path)
    dm.config.set("scenarios_folder", scenarios_tmp)
    # rx_sets에 실제 인덱스 범위를 전달해 메모리 절약
    # max_users>0 이면 첫 max_users 명의 UE만 로드
    rx_indices = list(range(max_users)) if max_users > 0 else "all"
    dataset = dm.load(
        scenario_name,
        tx_sets={tx_set_id: "all"},
        rx_sets={rx_set_id: rx_indices},
        max_paths=num_paths if num_paths > 0 else 10,
    )
    print(f"[preprocess] 로드 완료")

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
    # 채널이 0인 사용자(경로 없음) 제거
    ch_pow = np.abs(channels[:, 0, :, 0]).sum(axis=1)  # (N,) 첫 서브캐리어 전력 합
    valid_mask = ch_pow > 0
    n_valid = valid_mask.sum()
    print(f"[preprocess] 유효 사용자 수: {n_valid}/{n_users}")
    if n_valid < n_users:
        channels = channels[valid_mask]
        n_users = n_valid

    # 메모리 제한 시 사용자 수 축소
    if max_users > 0 and n_users > max_users:
        idx_sub = np.random.permutation(n_users)[:max_users]
        channels = channels[idx_sub]
        n_users = max_users
        print(f"[preprocess] 사용자 수 축소: {n_users}")

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
