import pytest
import torch

from src.model.gaussian_mlp_model import GaussianMLPModel

# helper function
def log_det_of_jacobian_of_tanh(x):
  return torch.log(1. - torch.tanh(x)*torch.tanh(x)).sum(dim=-1)

def log_det_of_jacobian_of_scale(scale, event_dim):
  return torch.log(torch.tensor([scale])**event_dim)


def test_init_using_classmethod():
  # expected to raise an assertion
  with pytest.raises(AssertionError):
    _ = GaussianMLPModel.instantiate_model(
      {"input_dim": 0,
       "output_dim": 0,
       "hidden_size": 0,
       "activation": None,
       "sample_scaling": None})
  
  # expected to success
  _ = GaussianMLPModel.instantiate_model(
    {"input_dim": 1,
       "output_dim": 1,
       "hidden_size": [],
       "activation": "relu",
       "sample_scaling": None})


def test_method_distribution():
  input_dim = 12
  output_dim = 24
  hidden_size = [36]

  model = GaussianMLPModel(
    input_dim=input_dim,
    output_dim=output_dim,
    hidden_size=hidden_size,
    activation="identity",
    sample_scaling=None
  )

  dist = model._distribution(torch.randn(1, 2, 3, input_dim))

  assert dist.batch_shape == (1, 2, 3)
  assert dist.event_shape == (output_dim,)


def test_method_forward():
  # sometimes life hits you with nan and inf
  # torch.manual_seed(10)

  input_dim = 8
  output_dim = 16
  hidden_size = [32]
  sample_scaling = 5.7
  batch_shape = (1, 2, 3)

  model = GaussianMLPModel(
    input_dim=input_dim,
    output_dim=output_dim,
    hidden_size=hidden_size,
    activation="identity",
    sample_scaling=sample_scaling
  )
  
  in_ = torch.randn(*batch_shape, input_dim)

  det_sample, det_log_prob = model(in_, True, True)
  stc_sample, stc_log_prob = model(in_, False, True)

  assert det_sample.shape == (*batch_shape, output_dim)
  assert stc_sample.shape == (*batch_shape, output_dim)
  assert det_log_prob.shape == batch_shape
  assert stc_log_prob.shape == batch_shape

  # compute the log_prob using change-of-variable formula
  # starting from the multivariate normal distribution
  mvn_dist = model._distribution(in_).base_dist
  det_sample = torch.atanh(det_sample / sample_scaling)
  stc_sample = torch.atanh(stc_sample / sample_scaling)

  det_log_prob_from_mvn = mvn_dist.log_prob(det_sample) \
    - log_det_of_jacobian_of_tanh(det_sample) \
    - log_det_of_jacobian_of_scale(sample_scaling, output_dim)
  stc_log_prob_from_mvn = mvn_dist.log_prob(stc_sample) \
    - log_det_of_jacobian_of_tanh(stc_sample) \
    - log_det_of_jacobian_of_scale(sample_scaling, output_dim)

  torch.allclose(det_log_prob, det_log_prob_from_mvn, atol=1e-2)
  torch.allclose(stc_log_prob, stc_log_prob_from_mvn, atol=1e-2)