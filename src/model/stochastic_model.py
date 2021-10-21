from typing import Any, Tuple

from torch import Tensor
from torch.distributions import Distribution

from src.model.model import Model

class StochasticModel(Model):
  """
  """
  def _distribution(self, x: Tensor) -> Distribution:
    raise NotImplementedError()

  """
  """
  def forward(self, x: Tensor, deterministic: bool, with_logprob: bool) -> Tuple[Tensor, Any]:
    raise NotImplementedError()

  """
  """
  def log_prob_from_data(self, x: Tensor, sample: Tensor) -> Tensor:
    distribution = self._distribution(x)
    return distribution.log_prob(sample)