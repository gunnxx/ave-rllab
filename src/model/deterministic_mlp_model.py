from typing import Dict, List, Type

import torch
import torch.nn as nn

from src.model.model import Model
from src.utils.common import create_mlp

class DeterministicMLPModel(Model):
  # List of constructor parameters
  # This will be used to check config in run.py
  REQUIRED_CONFIG_KEYS = {
    "input_dim": 1,
    "output_dim": 1,
    "hidden_size": [128],
    "activation": "relu",
    "output_activation": "softmax"
  }

  # Part of REQUIRED_CONFIG_KEYS to instantiate activation_fn
  ACTIVATION_CONFIG_KEYS = {
    "activation": None,
    "output_activation": None
  }

  """
  """
  def __init__(self,
    input_dim: int,
    output_dim: int,
    hidden_size: List[int],
    activation: Type[nn.Module],
    output_activation: Type[nn.Module]):
    super().__init__()

    hidden_size : List[int] = [input_dim] + hidden_size + [output_dim]
    self.network : nn.Sequential = create_mlp(
      hidden_size, activation, output_activation)
  
  """
  """
  @staticmethod
  def validate_params(params: Dict) -> None:
    pass

  """
  """
  def forward(self,
    x: torch.Tensor) -> torch.Tensor:
    return self.network(x)