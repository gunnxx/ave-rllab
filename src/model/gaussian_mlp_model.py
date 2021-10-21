from typing import Any, Dict, List, Tuple

from torch import Tensor, clamp, exp, tanh
from torch.nn import Linear, Sequential
from torch.distributions import Distribution, Normal, Independent, TransformedDistribution
from torch.distributions.transforms import AffineTransform, TanhTransform

from src.model.model import Model, REGISTERED_ACTIVATION_FUNC
from src.model.stochastic_model import StochasticModel
from src.utils.common import create_mlp

class GaussianMLPModel(StochasticModel):
  # Samples are drawn using reparameterization trick
  RSAMPLE = True

  # List of constructor parameters
  # This will be used to check config in run.py
  REQUIRED_CONFIG_KEYS = {
    "input_dim": 1,
    "output_dim": 1,
    "hidden_size": [128],
    "activation": "relu",
    "sample_scaling": None
  }

  """
  """
  def __init__(self,
    input_dim: int,
    output_dim: int,
    hidden_size: List[int],
    activation: str,
    sample_scaling: float = None) -> None:
    
    super().__init__(input_dim, output_dim)
    
    act = REGISTERED_ACTIVATION_FUNC[activation]
    hidden_size : List[int] = [input_dim] + hidden_size

    ## networks
    self.base_net : Sequential = create_mlp(hidden_size, act, act)
    self.mu_layer : Linear = Linear(hidden_size[-1], output_dim)
    self.log_std_layer : Linear = Linear(hidden_size[-1], output_dim)

    ## squash through tanh() then scale if sample_scaling != None
    if sample_scaling:
      self.transforms = [TanhTransform(), AffineTransform(0, sample_scaling)]
      self.transform_fn = lambda x: tanh(x)*sample_scaling
    
    ## do nothing if sampling_scale == None
    else:
      self.transforms = [AffineTransform(0, 1)]
      self.transform_fn = lambda x: x
  
  """
  Needs to make sure that `std > 0`.
  """
  def _distribution(self, x: Tensor) -> Distribution:
    base_net_out = self.base_net(x)

    ## compute the Gaussian parameters
    mu = self.mu_layer(base_net_out)
    log_std = self.log_std_layer(base_net_out)
    log_std = clamp(log_std, -20, 2)
    std = exp(log_std)

    dist = Independent(Normal(mu, std), 1)
    return TransformedDistribution(dist, self.transforms)

  """
  """
  def forward(self, x: Tensor, deterministic: bool, with_logprob: bool) -> Tuple[Tensor, Any]:
    ## get the Gaussian
    distribution = self._distribution(x)

    ## sampling
    if deterministic:
      sample = distribution.base_dist.base_dist.mean
      sample = self.transform_fn(sample)
    else:
      sample = distribution.rsample()

    ## compute log_prob
    log_prob_sample = distribution.log_prob(sample) if with_logprob else None

    return sample, log_prob_sample
  
  """
  """
  @staticmethod
  def validate_params(params: Dict) -> None:
    Model.validate_params(params)
    assert len(params["hidden_size"]) >= 0