from src.utils.common import cast_to_torch
from typing import Dict

import torch
import numpy as np

from src.buffer.buffer import Buffer
from src.utils.common import cast_to_torch

class RandomBuffer(Buffer):
  # List of constructor parameters
  # This will be used to check config in run.py
  REQUIRED_CONFIG_KEYS = {
    "buffer_size": 1e4
  }

  """
  """
  def __init__(self, buffer_size: int) -> None:
    super().__init__(buffer_size)
  
  """
  """
  @staticmethod
  def validate_params(params: Dict) -> None:
    assert params["buffer_size"] > 0

  """
  """
  def _store_mechanism(self, **kwargs) -> None:
    for k in self.data_keys:
      v = cast_to_torch(kwargs[k], torch.float32)

      try:
        self.data[k][self.ptr] = v
      except KeyError:
        raise KeyError("Current items to be stored \
          does not have {} key".format(k))
  
  """
  """
  def _sampler(self, batch_size: int) -> Dict[str, torch.Tensor]:
    idx = np.random.randint(0, self.size, size=batch_size)    
    return {k: v[idx] for k, v in self.data.items()}
    