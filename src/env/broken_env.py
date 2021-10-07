from typing import List, Union

import numpy as np
import torch

class BrokenEnv:
  """
  reversed_polarity:
    True  -> multiply the action by -1 (reverse)
    False -> multiply the action by +1 (normal)
  weakened_actuator:
    On the range of [0., 1.]
    0. -> dead actuator
    1. -> normal actuator
  """
  def __init__(self,
    n_actuators: int,
    reversed_polarity: List[bool],
    weakened_actuator: List[float]) -> None:

    self.n_actuators = n_actuators
    self.reversed_polarity = reversed_polarity
    self.weakened_actuator = weakened_actuator
  
  """
  """
  def set_reversed_polarity(self,
    reversed_polarity: List[bool]) -> None:
    assert len(reversed_polarity) == self.n_actuators
    self.reversed_polarity = reversed_polarity
  
  """
  """
  def set_weakened_actuator(self,
    weakened_actuator: List[float]) -> None:
    assert len(weakened_actuator) == self.n_actuators
    for val in weakened_actuator: assert 0. <= val <= 1.
    self.weakened_actuator = weakened_actuator
  
  """
  TO DO
  
  locked_actuator:
    True  -> the position of the actuator will be fixed to initial position
    False -> the position of the actuator will be able to move
  """
  # def set_locked_actuator(self,
  #   locked_actuator: List[bool]) -> None:
  #   assert len(locked_actuator) == 8
  #   self.locked_actuator = locked_actuator

  #   # lock the joint
  #   for i, lock in enumerate(locked_actuator):
  #     if lock:
  #       self.robot.ordered_joints[i].disable_motor()
  #       # self.robot.ordered_joints[i].power_coef = 0.0

  """
  TO DO

  misalign sensor reading e.g. displace joint position by 30deg
  """
  # def set_corrupt_sensor(self):
  #   pass

  """
  n_actuators:
    number of actuators to be randomly damaged
  n_damages:
    number of random damages to be applied to each n_actuators
    current version only support 2 types of damages
  """
  def randomize_broken_part(self,
    n_actuators: int,
    n_damages: int) -> None:
    assert 0 <= n_actuators <= self.n_actuators
    assert 0 <= n_damages <= 2

    # get which actuator and which damage to be applied to
    actuator_idx = np.random.choice(range(self.n_actuators), n_actuators, False)
    damage_idx = np.random.choice(range(2), n_damages, False)

    # just sample all damages to all actuators
    weakened_actuator = np.random.uniform(size=self.n_actuators)
    reversed_polarity = np.random.choice([True, False], size=self.n_actuators)

    # shut off damages to actuators that are not meant to be damaged
    for idx in range(self.n_actuators):
      if idx not in actuator_idx:
        weakened_actuator[idx] = 1.
        reversed_polarity[idx] = False
    
    # filter the damage type
    # current version only support 2 types of damages
    if damage_idx[0]:
      self.set_reversed_polarity(reversed_polarity)
    if damage_idx[1]:
      self.set_weakened_actuator(weakened_actuator)
  
  """
  """
  def apply_damage(self,
    a: Union[torch.Tensor, np.ndarray]) -> np.ndarray:
    act_modifier = np.array([-1 if i else 1 for i in self.reversed_polarity])
    act_modifier = act_modifier * self.weakened_actuator
    if type(a) == torch.Tensor: a = a.cpu().numpy()
    return a * act_modifier