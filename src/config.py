from typing import Any, Dict, Type

import os
import json
import gym

import src.env
import src.buffer as buffer
import src.model as model
from src.algo.algo import Algo

class Config:
  """
  Notes:
  Can't add additional keys besides what are specified
  in the Algo.REQUIRED_CONFIG_KEYS.
  """
  def __init__(self,
    algo_cls: Type[Algo]) -> None:
    self.algo_cls : Type[Algo] = algo_cls
    self.data : Dict = algo_cls.REQUIRED_CONFIG_KEYS

    ## Get the default params for model (algo-specific).
    for k, v in algo_cls.MODEL_CONFIG_KEYS.items():
      model_type = model.REGISTERED_MODEL[self.data[k]]
      self.data[v] = model_type.REQUIRED_CONFIG_KEYS
    
    ## Get the default params for buffer (algo-specific).
    for k, v in algo_cls.BUFFER_CONFIG_KEYS.items():
      buffer_type = buffer.REGISTERED_BUFFER[self.data[k]]
      self.data[v] = buffer_type.REQUIRED_CONFIG_KEYS
  
  """
  Override values of the config.
  """
  def fill(self, **kwargs) -> None:
    for k, v in kwargs.items():
      self.set(k, v)
  
  """
  Override values of the config from JSON file.
  """
  def fill_from_json(self, path: str) -> None:
    with open(path) as json_file:
      data = json.load(json_file)
      self.fill(**data)

  """
  Save current configuration as JSON file.
  """
  def save_as_json(self, path: str) -> None:
    assert not os.path.isdir(path), """exp_dir exists! change the
    exp_dir because we do not want to overwrite existing one."""

    os.makedirs(path)
    file_path = os.path.join(path, "config.json")
    with open(file_path, 'w') as json_file:
      json.dump(self.data, json_file, indent=4)
  
  """
  Set only when the key matches with what's already in
  the original config.
  """
  def set(self, key: str, val: Any) -> None:
    if key in self.data:
      self.data[key] = val
  
  """
  Map the non-json data-type.
  """
  def prepare(self) -> None:
    # env
    self.set(
      key="env",
      val=gym.make(self.data["env"]))
    
    # logger


    # buffer
    for buffer_type_key in self.algo_cls.BUFFER_CONFIG_KEYS:
      self.set(
        key=buffer_type_key,
        val=buffer.REGISTERED_BUFFER[self.data[buffer_type_key]])
    
    # model
    for model_type_key in self.algo_cls.MODEL_CONFIG_KEYS:
      self.set(
        key=model_type_key,
        val=model.REGISTERED_MODEL[self.data[model_type_key]])