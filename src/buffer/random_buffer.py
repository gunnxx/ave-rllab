from typing import List, Dict

import torch
import numpy as np

from src.buffer.buffer import Buffer
from src.utils.common import cast_to_torch

class RandomBuffer(Buffer):
  # List of constructor parameters
  # This will be used to check config in run.py
  REQUIRED_CONFIG_KEYS = {
    "buffer_size": int(1e4),
    "device": "cpu"
  }

  """
  """
  def __init__(self,
    buffer_size: int,
    device: str) -> None:
    super().__init__(buffer_size, device)
  
  """
  """
  @staticmethod
  def validate_params(params: Dict) -> None:
    assert params["buffer_size"] > 0

  """
  """
  def _store_mechanism(self, **kwargs) -> None:
    for k in self.data_keys:
      v = cast_to_torch(kwargs[k], torch.float32, self.device)

      try:
        self.data[k][self.ptr] = v
      except KeyError:
        raise KeyError("Current items to be stored \
          does not have {} key".format(k))
  
  """
  """
  def _idx_sampler(self, size: int) -> List[int]:
    idx = np.random.randint(0, self.size, size=size)
    return idx.tolist()
  
  """
  """
  def _consecutive_idx_sampler(self, size: int) -> List[int]:
    # when full, in low: max_size - size = 0
    # when not full, in low: ptr - size = 0
    low = self.max_size - self.size + self.ptr + size - 1
    high = self.max_size + self.ptr
    last_idx = np.random.randint(low, high)

    idx = np.arange(last_idx - size, last_idx)
    idx = (idx + 1) % self.max_size
    return idx.tolist()