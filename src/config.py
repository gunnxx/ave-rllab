from typing import Any, Dict, List, Union

import enum
import functools
import json
import operator

class VERBOSE(enum.Enum):
  SILENT = 1  # do not print anything
  UNUSED = 2  # print only unused keys
  ALL    = 3  # print unused and replaced keys

class Config:
  """
  Save all parameters passed.
  These will be default values from REQUIRED_CONFIG_KEYS.
  """
  def __init__(self,
    verbose: VERBOSE = VERBOSE.UNUSED,
    **kwargs) -> None:
    self.config : Dict = dict()
    self.verbose : VERBOSE = verbose

    for k, v in kwargs.items():
      self.config[k] = v
  
  """
  "One-layer" overriding values of the config.
  See set() to set value of a nested dictionary.
  """
  def override_config(self, **kwargs) -> None:
    for k, v in kwargs.items():
      if k in self.config: # REPLACE
        if self.verbose is VERBOSE.ALL:
          print("{:10s} is replaced.".format(k))
        self.config[k] = v
      
      else: # UNUSED
        if self.verbose is VERBOSE.UNUSED or \
            self.verbose is VERBOSE.ALL:
          print("{:10s} is unused. Possible typo?".format(k))
  
  """
  "One-layer" overriding values of the config from JSON file.
  """
  def override_config_from_json(self, path: str) -> None:
    with open(path) as json_file:
      data = json.load(json_file)
      self.override_config(data)

  """
  Save current configuration as JSON file.
  """
  def save_config_as_json(self, path: str) -> None:
    with open(path, 'w') as json_file:
      json.dump(self.config, json_file)
  

  """
  Set verbose to SILENT.
  """
  def silent(self) -> None:
    self.verbose = VERBOSE.SILENT
  
  """
  Allows to get values of a nested dictionary with list of keys.
  """
  def get(self, key: Union[str, List[str]]) -> Any:
    if type(key) is List:
      return functools.reduce(operator.getitem, key, self.config)
    return self.config.get(key, None)
  
  """
  Allows to set values of a nested dictionary with list of keys.
  """
  def set(self,
    key: Union[str, List[str]],
    val: Any) -> None:
    if type(key) is List:
      self.get(key[:-1])[key[-1]] = val
    else:
      self.config[key] = val
