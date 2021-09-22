from typing import Tuple, Union

import torch
import torch.distributions as distributions

from src.model.model import Model

class StochasticModel(Model):
  """
  """
  def _distribution(self,
    x: torch.Tensor) -> distributions.Distribution:
    raise NotImplementedError()

  """
  """
  def forward(self,
    x: torch.Tensor,
    deterministic: bool,
    with_logprob: bool
    ) -> Tuple[torch.Tensor, Union[torch.Tensor, None]]:
    raise NotImplementedError()

  """
  """
  def log_prob_from_data(self,
    x: torch.Tensor,
    sample: torch.Tensor) -> torch.Tensor:
    distribution = self._distribution(x)
    return distribution.log_prob(sample)