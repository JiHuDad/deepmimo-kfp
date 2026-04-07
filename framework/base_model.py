"""
[MLOps 제공] ModelInterface - 모델 개발자가 반드시 상속해야 하는 추상 클래스

사용법:
    from framework.base_model import ModelInterface

    class Model(ModelInterface):
        def build_model(self, input_dim, num_classes):
            ...
        def build_optimizer(self, model):
            ...

규칙:
  - 클래스 이름은 반드시 `Model` 이어야 합니다 (auto-discovery 규칙)
  - KFP, Kubeflow 관련 임포트는 일절 불필요합니다
  - PyTorch, numpy 등 순수 학습 코드만 작성하면 됩니다
"""

from abc import ABC, abstractmethod


class ModelInterface(ABC):
    """
    모든 모델이 구현해야 하는 인터페이스.

    __init__에서 self.config(dict)로 하이퍼파라미터에 접근할 수 있습니다.
    default_config()에 모델 기본 하이퍼파라미터를 선언하세요.
    """

    def __init__(self, config: dict):
        self.config = config

    # ── 필수 구현 ──────────────────────────────────────────────────

    @abstractmethod
    def build_model(self, input_dim: int, num_classes: int):
        """
        nn.Module 인스턴스를 반환하세요.

        Args:
            input_dim  : 입력 피처 차원 (전처리 결과에서 자동 결정)
            num_classes: 출력 클래스 수 (레이블에서 자동 결정)

        Returns:
            torch.nn.Module
        """

    @abstractmethod
    def build_optimizer(self, model):
        """
        (optimizer, scheduler) 튜플을 반환하세요.
        scheduler가 필요 없으면 (optimizer, None)을 반환하세요.

        Args:
            model: build_model()이 반환한 nn.Module

        Returns:
            tuple[Optimizer, LRScheduler | None]
        """

    # ── 선택 구현 ──────────────────────────────────────────────────

    @classmethod
    def default_config(cls) -> dict:
        """
        모델의 기본 하이퍼파라미터를 딕셔너리로 반환하세요.
        파이프라인 실행 시 이 값이 기본값으로 사용됩니다.

        Returns:
            dict  예: {"hidden_dims": "256,128,64", "dropout": 0.3}
        """
        return {}

    @property
    def extra_meta(self) -> dict:
        """
        model_meta.json에 추가로 저장할 정보를 반환하세요. (선택)
        모델 구조나 하이퍼파라미터 기록에 활용됩니다.
        """
        return {}
