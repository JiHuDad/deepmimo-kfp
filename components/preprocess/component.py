"""
preprocess 컴포넌트

DeepMIMO.generate_data()를 호출하여 레이트레이싱 기반 채널 행렬을 생성한다.
생성된 채널 데이터는 numpy 파일로 저장하고,
학습/검증/테스트 분리까지 수행한다.
"""

from kfp import dsl
from kfp.dsl import Input, Output, Dataset, Metrics


@dsl.component(
    base_image="192.168.1.112:5000/deepmimo-base:latest",
    packages_to_install=[],
)
def preprocess(
    scenario_dataset: Input[Dataset],
    parameters_json: str,
    train_ratio: float,
    val_ratio: float,
    output_train: Output[Dataset],
    output_val: Output[Dataset],
    output_test: Output[Dataset],
    output_metrics: Output[Metrics],
) -> None:
    """
    DeepMIMO 채널 생성 및 train/val/test 분리.

    parameters_json 예시:
    {
        "num_paths": 5,
        "active_BS": [1],
        "user_row_first": 1,
        "user_row_last": 100,
        "subcarriers": 512,
        "bandwidth": 0.5,
        "num_OFDM_subcarriers": 512,
        "OFDM_limit": 32
    }
    """
    import json
    import os

    import DeepMIMO
    import numpy as np

    # 파라미터 로드
    params = DeepMIMO.default_params()
    user_params = json.loads(parameters_json)
    params.update(user_params)
    params["dataset_folder"] = scenario_dataset.path

    print(f"[preprocess] DeepMIMO 파라미터: {params}")
    print(f"[preprocess] 시나리오 경로: {scenario_dataset.path}")

    # 채널 생성
    dataset = DeepMIMO.generate_data(params)

    # 채널 행렬 추출: shape (N_users, N_ant, N_subcarr, N_paths) -> 복소수
    channel = dataset[0]["user"]["channel"]
    n_users = channel.shape[0]
    print(f"[preprocess] 채널 shape: {channel.shape}, dtype: {channel.dtype}")

    # 빔 선택 레이블 생성 (각 사용자의 최적 빔 인덱스)
    # 간단 버전: 채널 크기가 가장 큰 안테나 인덱스를 레이블로 사용
    channel_power = np.abs(channel[:, :, 0, 0]) ** 2  # (N_users, N_ant)
    labels = np.argmax(channel_power, axis=1)          # (N_users,)

    # 특징 벡터: 채널 절대값 (실수 입력으로 변환)
    # shape: (N_users, N_ant * 2) — real/imag concat
    ch_flat = channel[:, :, 0, 0]  # (N_users, N_ant), 첫 번째 서브캐리어
    features = np.concatenate(
        [np.real(ch_flat), np.imag(ch_flat)], axis=1
    ).astype(np.float32)

    # train / val / test 분리
    idx = np.random.permutation(n_users)
    n_train = int(n_users * train_ratio)
    n_val = int(n_users * val_ratio)

    def _save_split(output: Output[Dataset], indices, name: str):
        os.makedirs(output.path, exist_ok=True)
        np.save(os.path.join(output.path, "features.npy"), features[indices])
        np.save(os.path.join(output.path, "labels.npy"), labels[indices])
        np.save(os.path.join(output.path, "channel.npy"), channel[indices])
        print(f"[preprocess] {name}: {len(indices)} 샘플 저장 → {output.path}")

    _save_split(output_train, idx[:n_train], "train")
    _save_split(output_val,   idx[n_train:n_train + n_val], "val")
    _save_split(output_test,  idx[n_train + n_val:], "test")

    # 메트릭 기록
    output_metrics.log_metric("total_users", n_users)
    output_metrics.log_metric("n_train", n_train)
    output_metrics.log_metric("n_val", n_val)
    output_metrics.log_metric("n_test", n_users - n_train - n_val)
    output_metrics.log_metric("n_bs_antennas", int(channel.shape[1]))
    output_metrics.log_metric("n_subcarriers", int(channel.shape[2]))
    output_metrics.log_metric("feature_dim", features.shape[1])
    output_metrics.log_metric("num_classes", int(labels.max()) + 1)
