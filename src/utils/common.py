from typing import Dict, List, Type

import torch.nn as nn

from config import Config

def create_mlp(
  hidden_size: List[int],
  activation: Type[nn.Module],
  output_activation: Type[nn.Module] = nn.Identity
  ) -> nn.Sequential:
  layers = []

  for i in range(len(hidden_size) - 1):
    act = activation if i < len(hidden_size)-2 else output_activation
    layers += [nn.Linear(hidden_size[i], hidden_size[i+1]), act()]
  
  return nn.Sequential(*layers)

def check_config_keys(
  config: Config,
  required_config_keys: List[str]) -> None:
  for key in required_config_keys:
    assert key in config, "{} is not present in the config".format(key)