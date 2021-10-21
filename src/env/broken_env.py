from typing import List, Union

import numpy as np
import torch

class BrokenEnv:
  """
  actuator_damage:
    Absolute value on the range of [0., 1.]
    0. -> dead actuator
    1. -> normal actuator "force"
    Sign of the value
    + -> normal
    - -> reverse movement
  """
  def __init__(self,
    n_actuators: int,
    actuator_damage: List[float]) -> None:
    self.n_actuators = n_actuators
    self.set_actuator_damage(actuator_damage)
  
  """
  """
  def set_actuator_damage(self,
    actuator_damage: List[float]) -> None:
    assert len(actuator_damage) == self.n_actuators
    for v in actuator_damage: assert -1 <= v <= 1
    self.actuator_damage = actuator_damage
  
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
    current version only support 3 types of damages;
    1) weakening  and 2) reverse polarity.
  """
  def randomize_damage(self,
    n_actuators: int,
    n_damages: int) -> None:
    assert 0 <= n_actuators <= self.n_actuators
    assert 0 <= n_damages <= 2

    # get which actuator and which damage to be applied to
    actuator_idx = np.random.choice(range(self.n_actuators), n_actuators, False)
    damage_idx = np.random.choice(range(2), n_damages, False)

    # just sample all damages to all actuators
    weakened_actuator = np.random.uniform(size=self.n_actuators)
    reversed_polarity = np.random.choice([1, -1], self.n_actuators)
    actuator_damage = np.ones(self.n_actuators)

    # create the damage mask
    for idx in range(self.n_actuators):
      if idx in actuator_idx:
        # filter the damage type
        if 0 in damage_idx:
          actuator_damage[idx] *= weakened_actuator[idx]
        if 1 in damage_idx:
          actuator_damage[idx] *= reversed_polarity[idx]
    
    # set the damage mask
    self.set_actuator_damage(actuator_damage)
  
  """
  """
  def apply_damage(self,
    a: Union[torch.Tensor, np.ndarray]) -> np.ndarray:
    if type(a) == torch.Tensor: a = a.cpu().numpy()
    return a * self.actuator_damage