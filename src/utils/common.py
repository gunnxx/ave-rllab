from typing import Any, List, Tuple, Type, Union

import torch
import torch.nn as nn
import numpy as np

from PIL import Image
from PIL.ImageDraw import Draw

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
  data: Union[torch.Tensor, np.ndarray, List[Any]],
  dtype: torch.dtype,
  device: torch.device) -> torch.Tensor:
  
  if type(data) is not torch.Tensor:
    return torch.tensor(data, dtype=dtype, device=device)
  
  elif data.dtype is not dtype:
    return data.type(dtype).to(device)
  
  return data.to(device)

"""
"""
def batch_data(*args):
  data = []
  for d in args:
    data.append(torch.unsqueeze(d, dim=0))
  return torch.cat(data)

"""
"""
def label_frame(
  frame: np.ndarray,
  text_color: Tuple[int] = (0, 0, 0),
  **kwargs) -> Image.Image:
  im = Image.fromarray(frame)
  drawer = Draw(im)
  
  x_loc = im.size[0]/20
  y_loc = im.size[1]/10

  for k, v in kwargs.items():
    text = str(k) + " : " + str(v)
    drawer.text((x_loc, y_loc), text, fill=text_color)
    y_loc += 15
  
  return im

"""
"""
def warn_and_ask(text: str) -> None:
  ans = input(text)
  if ans.lower() == 'y': pass
  elif ans.lower() == 'n': raise AssertionError("Terminated due to warning.")
  else: raise KeyError("Answer is not recognized.")