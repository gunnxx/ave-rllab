from typing import Dict, List

import torch
import torch.nn as nn

from src.model.model import Model, REGISTERED_ACTIVATION_FUNC
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
  
  """
  """
  def __init__(self,
    input_dim: int,
    output_dim: int,
    hidden_size: List[int],
    activation: str,
    output_activation: str):
    super().__init__()

    activation = REGISTERED_ACTIVATION_FUNC[activation]
    output_activation = REGISTERED_ACTIVATION_FUNC[output_activation]

    hidden_size : List[int] = [input_dim] + hidden_size + [output_dim]
    self.network : nn.Sequential = create_mlp(
      hidden_size, activation, output_activation)
  
  """
  """
  @staticmethod
  def validate_params(params: Dict) -> None:
    assert params["input_dim"] > 0
    assert params["output_dim"] > 0
    assert len(params["hidden_size"]) >= 0

  """
  """
  def forward(self,
    x: torch.Tensor) -> torch.Tensor:
    return self.network(x)