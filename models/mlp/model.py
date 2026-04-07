"""
[모델 개발자 작성] MLP 빔 선택 모델

작성 규칙:
  1. 클래스 이름은 반드시 `Model` 이어야 합니다
  2. framework.base_model.ModelInterface 를 상속하세요
  3. build_model(), build_optimizer() 를 구현하세요
  4. KFP, Kubeflow 관련 코드는 일절 불필요합니다

하이퍼파라미터는 default_config()에 선언하고,
실행 시 self.config["키이름"] 으로 접근합니다.
"""

import torch.nn as nn
import torch.optim as optim

from framework.base_model import ModelInterface


class Model(ModelInterface):

    @classmethod
    def default_config(cls) -> dict:
        return {
            "hidden_dims": "256,128,64",
            "dropout": 0.3,
            "learning_rate": 0.001,
        }

    def build_model(self, input_dim: int, num_classes: int) -> nn.Module:
        hidden_dims = self.config.get("hidden_dims", "256,128,64")
        dropout     = float(self.config.get("dropout", 0.3))

        dims = [input_dim] + [int(d) for d in hidden_dims.split(",")] + [num_classes]
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(nn.BatchNorm1d(dims[i + 1]))
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(dropout))
        return nn.Sequential(*layers)

    def build_optimizer(self, model: nn.Module):
        lr = float(self.config.get("learning_rate", 0.001))
        optimizer = optim.Adam(model.parameters(), lr=lr)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=5, factor=0.5
        )
        return optimizer, scheduler

    @property
    def extra_meta(self) -> dict:
        return {
            "hidden_dims": self.config.get("hidden_dims"),
            "dropout": self.config.get("dropout"),
        }
