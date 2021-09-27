from typing import Dict, List, Type

import torch

from src.utils.common import cast_to_torch

class Buffer:
  """
  """
  def __init__(self,
    buffer_size: int,
    device: str) -> None:
    self.max_size : int = buffer_size
    self.ptr : int = 0
    self.size : int = 0
    
    self.data : Dict[str, torch.Tensor] = dict()
    self.data_keys : List[str] = list()
    
    self.device : torch.device = torch.device(device)

  """
  """
  def _init_buffer(self, **kwargs) -> None:
    for k, v in kwargs.items():
      v = cast_to_torch(v, torch.float32, self.device)
      self.data[k] = torch.zeros(
        (self.max_size, *v.shape),
        dtype=torch.float32).to(self.device)
      
      self.data_keys.append(k)

  """
  """
  def _store_mechanism(self, **kwargs) -> None:
    raise NotImplementedError()

  """
  """
  def _sampler(self, batch_size: int) -> Dict[str, torch.Tensor]:
    raise NotImplementedError()

  """
  """
  def store(self, **kwargs) -> None:
    if self.data_keys == list():
      self._init_buffer(**kwargs)

    self._store_mechanism(**kwargs)

    self.ptr = (self.ptr + 1) % self.max_size
    self.size = min(self.size + 1, self.max_size)
  
  """
  """
  def sample_batch(self, batch_size: int) -> Dict[str, torch.Tensor]:
    return self._sampler(batch_size)
  
  """
  """
  def sample_idx(self, idx: List[int]) -> Dict[str, torch.Tensor]:
    return {k:v[idx] for k, v in self.data.items()}

  """
  """
  def sample_last_n(self, n: int) -> Dict[str, torch.Tensor]:
    assert n <= self.size, "The buffer does not have >=n elements"
    idx = list(range(self.ptr - n, self.ptr))
    return self.sample_idx(idx)
  
  """
  """
  @classmethod
  def instantiate_buffer(
    buffer_type: "Type[Buffer]",
    buffer_params: Dict) -> "Buffer":
    
    buffer_type.validate_params(buffer_params)  
    return buffer_type(**buffer_params)