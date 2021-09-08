from typing import List, Tuple, Type, Union

import numpy as np
import torch
import torch.nn as nn
import torch.distributions as distributions
import torch.nn.functional as F

from stochastic_model import StochasticModel
from utils.common import create_mlp

class SquashedGaussianMLPModel(StochasticModel):
  # Samples are drawn using reparameterization trick
  RSAMPLE = True

  # List of constructor parameters
  # This will be used to check config in run.py
  REQUIRED_CONFIG_KEYS = [
    "input_dim",
    "output_dim",
    "hidden_size",
    "activation",
    "sample_scaling"
  ]

  # Part of REQUIRED_CONFIG_KEYS to instantiate activation_fn
  ACTIVATION_CONFIG_KEYS = [
    "activation"
  ]

  def __init__(self,
    input_dim: int,
    output_dim: int,
    hidden_size: List[int],
    activation: Type[nn.Module],
    sample_scaling: float = 1.) -> None:
    super().__init__()

    hidden_size = [input_dim] + hidden_size
    self.base_network = create_mlp(hidden_size, activation, activation)
    self.mu_layer = nn.Linear(hidden_size[-1], output_dim)
    self.log_std_layer = nn.Linear(hidden_size[-1], output_dim)

    self.sample_scaling = sample_scaling
  
  def _distribution(self,
    x: torch.Tensor) -> distributions.Distribution:
    base_net_out = self.base_network(x)

    mu = self.mu_layer(base_net_out)
    std = torch.exp(self.log_std_layer(base_net_out))

    return distributions.Normal(mu, std)

  def _log_prob_from_distribution(self,
    distribution: distributions.Distribution,
    sample: torch.Tensor) -> torch.Tensor:
    # had to take into account change-of-variable formula
    log_prob_sample = distribution.log_prob(sample).sum(axis=-1)
    cov_term = (2*(np.log(2) - sample - F.softplus(-2*sample))).sum(axis=1)
    return log_prob_sample + cov_term
  
  def forward(self,
    x: torch.Tensor,
    sample: torch.Tensor,
    deterministic: bool,
    with_logprob: bool
    ) -> Tuple[torch.Tensor, Union[torch.Tensor, None]]:
    distribution = self._distribution(x)

    # sampling
    if deterministic: sample = distribution.mean
    else: sample = distribution.rsample()

    # compute log prob
    # need to take into account squashing function tanh()
    log_prob_sample = self._log_prob_from_distribution(
      distribution, sample) if with_logprob else None
    
    # squash the sample
    sample = torch.tanh(sample)
    sample = sample * self.sample_scaling

    return sample, log_prob_sample
