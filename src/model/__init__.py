import torch.nn as nn

from model.deterministic_mlp_model import DeterministicMLPModel
from model.squashed_gaussian_mlp_model import SquashedGaussianMLPModel
from torch.nn.modules.activation import Sigmoid

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