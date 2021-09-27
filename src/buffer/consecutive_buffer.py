from typing import Dict, List, Union

import torch
import numpy as np

from src.buffer.buffer import Buffer
from src.utils.common import cast_to_torch

class ConsecutiveBuffer(Buffer):
  # List of constructor parameters
  # This will be used to check config in run.py
  REQUIRED_CONFIG_KEYS = {
    "buffer_size": int(1e4),
    "consecutive_size": 20,
    "device": "cpu"
  }

  """
  """
  def __init__(self,
    buffer_size: int,
    consecutive_size: int,
    device: str) -> None:
    super().__init__(buffer_size, device)

    self.queue : Dict[str, List[torch.Tensor]] = {}
    self.consecutive_size : int = consecutive_size
  
  """
  """
  @staticmethod
  def validate_params(params: Dict) -> None:
    assert params["buffer_size"] > 0
    assert params["buffer_size"] >= params["consecutive_size"]

  """
  """
  def _init_buffer(self, **kwargs) -> None:
    for k, v in kwargs.items():
      v = cast_to_torch(v, torch.float32, self.device)

      self.data[k] = torch.zeros(
        (self.max_size, self.consecutive_size, *v.shape),
        dtype=torch.float32).to(self.device)
      
      self.queue[k] = []
      self.data_keys.append(k)
  
  """
  """
  def _store_mechanism(self, **kwargs) -> None:
    for k in self.data_keys:
      # single value
      v = cast_to_torch(kwargs[k], torch.float32, self.device)

      # put it on queue to join it with previous values
      self.queue[k].append(torch.unsqueeze(v, dim=0))
      if len(self.queue[k]) > self.consecutive_size:
        self.queue[k].pop(0)

      # store on buffer
      if len(self.queue[k]) == self.consecutive_size:
        try:
          self.data[k][self.ptr] = torch.cat(self.queue[k])
        except KeyError:
          raise KeyError("Current items to be stored \
            does not have {} key".format(k))
      
      # not yet storing because the queue has not reached the
      # consecutive_size, so we decrease counter because every
      # calls to this function increases the counter by default
      else:
        self.ptr -= 1
        self.size -= 1
  
  """
  """
  def _sampler(self, batch_size: int) -> Dict[str, torch.Tensor]:
    idx = np.random.randint(0, self.size, size=batch_size)    
    return {k: v[idx] for k, v in self.data.items()}