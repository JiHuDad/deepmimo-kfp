"""
[모델 개발자 작성] Transformer 빔 선택 모델

작성 규칙:
  1. 클래스 이름은 반드시 `Model` 이어야 합니다
  2. framework.base_model.ModelInterface 를 상속하세요
  3. build_model(), build_optimizer() 를 구현하세요
  4. KFP, Kubeflow 관련 코드는 일절 불필요합니다
"""

import math

import torch
import torch.nn as nn
import torch.optim as optim

from framework.base_model import ModelInterface


class Model(ModelInterface):

    @classmethod
    def default_config(cls) -> dict:
        return {
            "d_model": 64,
            "num_heads": 4,
            "num_layers": 2,
            "dropout": 0.1,
            "learning_rate": 0.0005,
        }

    def build_model(self, input_dim: int, num_classes: int) -> nn.Module:
        d_model    = int(self.config.get("d_model", 64))
        num_heads  = int(self.config.get("num_heads", 4))
        num_layers = int(self.config.get("num_layers", 2))
        dropout    = float(self.config.get("dropout", 0.1))

        seq_len    = math.ceil(input_dim / d_model)
        padded_dim = seq_len * d_model

        class BeamTransformer(nn.Module):
            def __init__(self):
                super().__init__()
                self.input_proj = nn.Linear(input_dim, padded_dim)
                encoder_layer = nn.TransformerEncoderLayer(
                    d_model=d_model,
                    nhead=num_heads,
                    dim_feedforward=d_model * 4,
                    dropout=dropout,
                    batch_first=True,
                )
                self.encoder    = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
                self.classifier = nn.Sequential(
                    nn.LayerNorm(d_model),
                    nn.Linear(d_model, num_classes),
                )
                self._seq_len = seq_len
                self._d_model = d_model

            def forward(self, x):
                x = self.input_proj(x)
                x = x.view(x.size(0), self._seq_len, self._d_model)
                x = self.encoder(x)
                x = x.mean(dim=1)
                return self.classifier(x)

        return BeamTransformer()

    def build_optimizer(self, model: nn.Module):
        lr = float(self.config.get("learning_rate", 0.0005))
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
        # num_epochs를 모르므로 T_max는 config에서 받거나 기본값 사용
        t_max     = int(self.config.get("num_epochs", 50))
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=t_max)
        return optimizer, scheduler

    @property
    def extra_meta(self) -> dict:
        return {
            "d_model":    self.config.get("d_model"),
            "num_heads":  self.config.get("num_heads"),
            "num_layers": self.config.get("num_layers"),
        }
