from src.utils.common import create_mlp
from typing import List, Type

import torch
import torch.nn as nn

from model import Model

class DeterministicMLPModel(Model):
  # List of constructor parameters
  # This will be used to check config in run.py
  REQUIRED_CONFIG_KEYS = [
    "input_dim",
    "output_dim",
    "hidden_size",
    "activation",
    "output_activation"
  ]

  # Part of REQUIRED_CONFIG_KEYS to instantiate activation_fn
  ACTIVATION_CONFIG_KEYS = [
    "activation",
    "output_activation"
  ]

  def __init__(self,
    input_dim: int,
    output_dim: int,
    hidden_size: List[int],
    activation: Type[nn.Module],
    output_activation: Type[nn.Module]):
    super().__init__()

    hidden_size = [input_dim] + hidden_size + [output_dim]
    self.network = create_mlp(
      hidden_size, activation, output_activation)

  def forward(self,
    x: torch.Tensor) -> torch.Tensor:
    return self.network(x)