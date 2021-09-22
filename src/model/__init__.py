import torch.nn as nn

from src.model.deterministic_mlp_model import DeterministicMLPModel
from src.model.gaussian_mlp_model import GaussianMLPModel

REGISTERED_MODEL = {
  'deterministic_mlp_model': DeterministicMLPModel,
  'gaussian_mlp_model': GaussianMLPModel
}