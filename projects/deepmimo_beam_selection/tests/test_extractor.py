"""
features/extractor.py 단위 테스트.

폐쇄망 환경에서도 실행 가능하도록 numpy 만 사용.
KFP / DeepMIMO / PyTorch 설치 불필요.

실행:
    cd /home/user/deepmimo-kfp
    python -m pytest projects/deepmimo_beam_selection/tests/test_extractor.py -v
"""

import numpy as np
import pytest

from projects.deepmimo_beam_selection.features import (
    extract_features,
    extract_labels,
    filter_valid_users,
    feature_dim,
    schema,
    DEFAULT_SUBCARRIER_INDEX,
    DEFAULT_RX_ANTENNA_INDEX,
)


# ── Fixtures ─────────────────────────────────────────────────────────

@pytest.fixture
def channels_8tx():
    """N=20, rx=1, tx=8, subc=4 형태의 가짜 채널."""
    rng = np.random.default_rng(42)
    real = rng.standard_normal((20, 1, 8, 4)).astype(np.float32)
    imag = rng.standard_normal((20, 1, 8, 4)).astype(np.float32)
    return real + 1j * imag


@pytest.fixture
def channels_with_zeros(channels_8tx):
    """일부 사용자의 채널을 0으로 만든 fixtures."""
    ch = channels_8tx.copy()
    ch[3]  = 0.0   # 3번 사용자 무효
    ch[11] = 0.0   # 11번 사용자 무효
    return ch


# ── extract_features ─────────────────────────────────────────────────

class TestExtractFeatures:

    def test_shape(self, channels_8tx):
        feats = extract_features(channels_8tx)
        assert feats.shape == (20, 16), f"expected (20,16), got {feats.shape}"

    def test_dtype_float32(self, channels_8tx):
        feats = extract_features(channels_8tx)
        assert feats.dtype == np.float32

    def test_real_imag_split(self, channels_8tx):
        """앞 절반이 real, 뒤 절반이 imag 인지 확인."""
        feats = extract_features(channels_8tx)
        n_tx = channels_8tx.shape[2]
        ch_slice = channels_8tx[:, DEFAULT_RX_ANTENNA_INDEX, :, DEFAULT_SUBCARRIER_INDEX]
        np.testing.assert_allclose(feats[:, :n_tx], np.real(ch_slice), rtol=1e-5)
        np.testing.assert_allclose(feats[:, n_tx:], np.imag(ch_slice), rtol=1e-5)

    def test_custom_indices(self, channels_8tx):
        """서브캐리어·안테나 인덱스 파라미터가 동작하는지 확인."""
        feats = extract_features(channels_8tx, subcarrier_index=2, rx_antenna_index=0)
        n_tx = channels_8tx.shape[2]
        ch_slice = channels_8tx[:, 0, :, 2]
        np.testing.assert_allclose(feats[:, :n_tx], np.real(ch_slice), rtol=1e-5)

    def test_invalid_dim_raises(self):
        with pytest.raises(ValueError, match="4차원"):
            extract_features(np.zeros((5, 8)))          # 2D

    def test_invalid_dim_3d_raises(self):
        with pytest.raises(ValueError, match="4차원"):
            extract_features(np.zeros((5, 1, 8)))       # 3D


# ── extract_labels ────────────────────────────────────────────────────

class TestExtractLabels:

    def test_shape(self, channels_8tx):
        labels = extract_labels(channels_8tx)
        assert labels.shape == (20,)

    def test_dtype_int64(self, channels_8tx):
        labels = extract_labels(channels_8tx)
        assert labels.dtype == np.int64

    def test_range(self, channels_8tx):
        labels = extract_labels(channels_8tx)
        assert labels.min() >= 0
        assert labels.max() < channels_8tx.shape[2]   # < n_tx

    def test_correctness_manual(self):
        """명확한 정답이 있는 채널로 argmax 결과 검증."""
        ch = np.zeros((3, 1, 4, 1), dtype=np.complex64)
        ch[0, 0, 1, 0] = 10 + 0j   # 사용자 0 → 빔 1
        ch[1, 0, 3, 0] =  5 + 5j   # 사용자 1 → 빔 3  (|5+5j|^2 = 50)
        ch[2, 0, 0, 0] =  1 + 0j   # 사용자 2 → 빔 0
        labels = extract_labels(ch)
        np.testing.assert_array_equal(labels, [1, 3, 0])

    def test_invalid_dim_raises(self):
        with pytest.raises(ValueError, match="4차원"):
            extract_labels(np.zeros((5, 8)))


# ── filter_valid_users ────────────────────────────────────────────────

class TestFilterValidUsers:

    def test_all_valid(self, channels_8tx):
        mask = filter_valid_users(channels_8tx)
        assert mask.all(), "모든 사용자가 유효해야 함"
        assert mask.shape == (20,)

    def test_zero_users_filtered(self, channels_with_zeros):
        mask = filter_valid_users(channels_with_zeros)
        assert mask.shape == (20,)
        assert not mask[3],  "3번 사용자는 무효여야 함"
        assert not mask[11], "11번 사용자는 무효여야 함"
        assert mask.sum() == 18

    def test_invalid_dim_raises(self):
        with pytest.raises(ValueError, match="4차원"):
            filter_valid_users(np.zeros((5, 8)))


# ── feature_dim / schema ──────────────────────────────────────────────

class TestHelpers:

    def test_feature_dim(self):
        assert feature_dim(8)  == 16
        assert feature_dim(64) == 128
        assert feature_dim(1)  == 2

    def test_schema_has_version(self):
        s = schema()
        assert "version" in s
        assert s["version"] == "1.0.0"

    def test_schema_consistent_with_defaults(self):
        s = schema()
        assert s["subcarrier_index"] == DEFAULT_SUBCARRIER_INDEX
        assert s["rx_antenna_index"] == DEFAULT_RX_ANTENNA_INDEX

    def test_schema_has_required_keys(self):
        required = {"version", "subcarrier_index", "rx_antenna_index",
                    "feature_format", "label_policy"}
        assert required <= set(schema().keys())


# ── 학습-서빙 일관성 (regression) ────────────────────────────────────

class TestRegressionConsistency:
    """
    같은 입력에 항상 같은 출력을 보장하는 regression 테스트.
    extractor.py 를 수정했을 때 이 테스트가 깨지면 모델 재학습이 필요하다.
    """

    def test_features_deterministic(self, channels_8tx):
        f1 = extract_features(channels_8tx)
        f2 = extract_features(channels_8tx)
        np.testing.assert_array_equal(f1, f2)

    def test_labels_deterministic(self, channels_8tx):
        l1 = extract_labels(channels_8tx)
        l2 = extract_labels(channels_8tx)
        np.testing.assert_array_equal(l1, l2)

    def test_features_fixed_seed_snapshot(self):
        """고정 시드로 생성한 채널에서의 feature 첫 행 고정값 테스트.
        extractor 정책이 바뀌면 이 값도 바뀐다.
        """
        rng = np.random.default_rng(0)
        ch = (rng.standard_normal((1, 1, 4, 2))
              + 1j * rng.standard_normal((1, 1, 4, 2))).astype(np.complex64)
        feats = extract_features(ch)            # (1, 8)
        expected_real = np.real(ch[0, 0, :, 0])
        expected_imag = np.imag(ch[0, 0, :, 0])
        np.testing.assert_allclose(feats[0, :4], expected_real, rtol=1e-5)
        np.testing.assert_allclose(feats[0, 4:], expected_imag, rtol=1e-5)
