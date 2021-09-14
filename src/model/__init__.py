import torch.nn as nn

from src.model.deterministic_mlp_model import DeterministicMLPModel
from src.model.squashed_gaussian_mlp_model import SquashedGaussianMLPModel

REGISTERED_MODEL = {
  'deterministic_mlp_model': DeterministicMLPModel,
  'squashed_gaussian_mlp_model': SquashedGaussianMLPModel
}

REGISTERED_ACTIVATION_FUNC = {
  "tanh": nn.Tanh,
  "relu": nn.ReLU,
  "sigmoid": nn.Sigmoid,
  "softmax": nn.Softmax
}