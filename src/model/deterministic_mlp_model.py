from typing import Dict, List

from torch import Tensor
from torch.nn import Sequential

from src.model.model import Model, REGISTERED_ACTIVATION_FUNC
from src.model.deterministic_model import DeterministicModel
from src.utils.common import create_mlp

class DeterministicMLPModel(DeterministicModel):
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
    output_activation: str) -> None:
    
    super().__init__(input_dim, output_dim)

    activation = REGISTERED_ACTIVATION_FUNC[activation]
    output_activation = REGISTERED_ACTIVATION_FUNC[output_activation]
    hidden_size : List[int] = [input_dim] + hidden_size + [output_dim]

    ## networks
    self.network : Sequential = create_mlp(hidden_size, activation, output_activation)
  
  """
  """
  def forward(self, x: Tensor) -> Tensor:
    return self.network(x)
  
  """
  """
  @staticmethod
  def validate_params(params: Dict) -> None:
    Model.validate_params(params)
    assert len(params["hidden_size"]) >= 0