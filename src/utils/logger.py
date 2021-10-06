from typing import Any, Dict, List

import os
import numpy as np
import torch

class Logger:
  """
  Support two methods of logging:
  [1] Epoch logging:
  {
    "reward": [],
    "q_loss": [],
    "episode_length": []
  }
  Its mean, min, max, and std will be logged to `log.epoch.tsv`
  after each call to `dump()`. What is logged will be determined
  during the first dump. Adding new keys to be dumped on the
  second dump will not work.

  [2] (Ordinary) logging:
  {
    "reward": {
      "reward": [],
      "timestep": []
    },
    "q_loss": {
      "q1_loss": [],
      "q2_loss": [],
      "timestep": []
    }
  }
  It will be logged to `log.reward.tsv` and `log.q_loss.tsv`. It has
  to be separated in case reward and q_loss have different lengths.
  """
  def __init__(self,
    exp_dir: str,
    spacing: int = 15,
    float_precision: int = 6) -> None:
    os.makedirs(os.path.join(exp_dir, "torch"))

    self.exp_dir = exp_dir
    self.spacing = spacing
    self.float_precision = float_precision

    ## `first_dump` and `log_header` are needed to ensure
    ## periodic dumping of `epoch_log` is uniform (e.g. no
    ## additional key insertion in the middle of training).
    ## `column_name` is just `log_header` with additional
    ## details e.g. min, max, etc.
    self.first_dump : bool = True
    self.log_header : List = list()
    self.column_name : List = list()

    ## difference between `log` and `epoch_log` is that 
    ## `epoch_log` will be periodically freed and its mean,
    ## max, min, and std will be computed.
    self.log : Dict = dict()
    self.epoch_log : Dict = dict()
    
    ## corresponding output file for `log` and `epoch_log`.
    ## `log` might has variable length data, so it will treat
    ## each key as its own separate tsv files.
    self.output = dict()
    self.epoch_output = open(
      os.path.join(self.exp_dir, "log.epoch.tsv"), "w")
  
  """
  Helper function to join list of values (int, float, str).
  """
  def _join(self, vals: List[Any]) -> str:
    res = ""
    for val in vals:
      if type(val) == float: # float
        res += "{:<{spacing}.{precision}f} ".format(
          val, spacing=self.spacing, precision=self.float_precision)
      else: # int and str
        res += "{:{spacing}s} ".format(
          str(val), spacing=self.spacing)
    return res

  """
  Dump `self.epoch_log` so it can be freed for next epoch.
  The mean, max, min, and std will be computed.
  """
  def _dump_epoch_log(self) -> Dict[str, float]:
    if self.first_dump:
      ## save log_header
      self.log_header = list(self.epoch_log.keys())
      self.first_dump = False

      ## write the column name on top of the log file
      for k in self.log_header:
        if len(self.epoch_log[k]) > 1:
          self.column_name += [k+"-avg", k+"-min", k+"-max", k+"-std"]
        else:
          self.column_name += [k]
      self.epoch_output.write(self._join(self.column_name) + "\n")
    
    ## compute mean, min, max, std if it is a list > 1 elements.
    vals = []
    for k in self.log_header:
      val = self.epoch_log.get(k, [None])
      if len(val) > 1:
        vals.append(np.average(val).item())
        vals.append(min(val))
        vals.append(max(val))
        vals.append(np.std(val).item())
      else:
        vals.append(val[0])
    
    self.epoch_output.write(self._join(vals) + "\n")
    self.epoch_output.flush()
    self.epoch_log.clear()

    return {k: v for k, v in zip(self.column_name, vals)}

  """
  Dump `self.log` to save memory and save progress.
  """
  def _dump_ordinary_log(self) -> None:
    for k, vals in self.log.items():
      ## first dump
      if k not in self.output:
        self.output[k] = open(
          os.path.join(self.exp_dir, "log." + k + ".tsv"), "w")
        self.output[k].write(self._join(vals.keys()) + "\n")
      
      ## write logs
      for v in zip(*vals.values()):
        self.output[k].write(self._join(v) + "\n")
      
      self.output[k].flush()
    self.log.clear()

  """
  """
  def dump(self) -> Dict[str, float]:
    self._dump_ordinary_log()
    epoch_stat = self._dump_epoch_log()
    return epoch_stat
  
  """
  (Ordinary) logging. See comment on top for illustration.
  """
  def store(self,
    **kwargs: Dict[str, Dict]) -> None:
    for master_k, vals in kwargs.items():
      if master_k not in self.log:
        self.log[master_k] = {}
      
      for k, v in vals.items():
        if k not in self.log[master_k]:
          self.log[master_k][k] = []
        self.log[master_k][k].append(v)
  
  """
  Epoch logging. See comment on top for illustration.
  """
  def epoch_store(self, **kwargs) -> None:
    for k, v in kwargs.items():
      if k not in self.epoch_log:
        self.epoch_log[k] = []
      self.epoch_log[k].append(v)

  """
  Call to `torch.save()` for convenience only because we dont
  have to specify `exp_dir` when calling `logger.torch_save()`.
  """
  def torch_save(self,
    objects: Dict,
    filename: str) -> None:
    torch.save(
      objects,
      os.path.join(self.exp_dir, "torch", filename))
  
  """
  Close all files and generate plot from dumped
  `self.log` and `self.epoch_log` if needed.
  """
  def close(self):
    self.epoch_output.close()
    for _, f in self.output.items(): f.close()