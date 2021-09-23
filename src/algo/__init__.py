import torch.optim as optim
from src.algo.grbal import GrBAL

REGISTERED_ALGO = {
  "original_grbal": GrBAL
}

REGISTERED_OPTIM = {
  "adam": optim.Adam,
  "sgd": optim.SGD
}