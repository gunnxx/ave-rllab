from torch import Tensor
from src.model.model import Model

class DeterministicModel(Model):
  """
  """
  def forward(self, x: Tensor) -> Tensor:
    raise NotImplementedError