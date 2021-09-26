from typing import Dict, List, Type, Union

import torch
import torch.nn as nn
import numpy as np

"""
"""
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

"""
"""
def cast_to_torch(
  data: Union[torch.Tensor, np.array],
  dtype: torch.dtype) -> torch.Tensor:
  
  if type(data) is not torch.Tensor:
    return torch.tensor(data, dtype=dtype)
  
  elif data.dtype is not dtype:
    return data.type(dtype)
  
  return data

"""
"""
def batch_data(*args):
  data = []
  for d in args:
    data.append(torch.unsqueeze(d, dim=0))
  return torch.cat(data)