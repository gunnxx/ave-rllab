import pytest
import torch

from src.model import REGISTERED_ACTIVATION_FUNC
from src.model.deterministic_mlp_model import DeterministicMLPModel

def test_init_using_classmethod():
  # expected to raise an assertion
  with pytest.raises(AssertionError):
    _ = DeterministicMLPModel.instantiate_model(
      {"input_dim": 0,
       "output_dim": 0,
       "hidden_size": 0,
       "activation": None,
       "output_activation": None})
  
  # expected to success
  _ = DeterministicMLPModel.instantiate_model(
    {"input_dim": 1,
     "output_dim": 1,
     "hidden_size": [],
     "activation": REGISTERED_ACTIVATION_FUNC["relu"],
     "output_activation": REGISTERED_ACTIVATION_FUNC["identity"]})


def test_method_forward():
  input_dim  = 12
  output_dim = 24
  hidden_size = [36]

  model = DeterministicMLPModel(
    input_dim=input_dim,
    output_dim=output_dim,
    hidden_size=hidden_size,
    activation=REGISTERED_ACTIVATION_FUNC["identity"],
    output_activation=REGISTERED_ACTIVATION_FUNC["identity"]
  )

  out = model(torch.randn(1, 2, 3, input_dim))

  assert out.shape == (1, 2, 3, output_dim)
