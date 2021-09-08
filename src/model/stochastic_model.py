from typing import Tuple, Union

from model import Model

import torch
import torch.distributions as distributions

class StochasticModel(Model):
  def _distribution(self,
    x: torch.Tensor) -> distributions.Distribution:
    raise NotImplementedError
  
  def _log_prob_from_distribution(self,
    distribution: distributions.Distribution,
    sample: torch.Tensor) -> torch.Tensor:
    raise NotImplementedError

  def forward(self,
    x: torch.Tensor,
    sample: torch.Tensor,
    deterministic: bool,
    with_logprob: bool
    ) -> Tuple[torch.Tensor, Union[torch.Tensor, None]]:
    raise NotImplementedError