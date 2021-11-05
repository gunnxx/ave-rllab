from typing import Dict, List

import gym.spaces
import numpy as np
from pybulletgym.envs.mujoco.envs.locomotion.ant_env import AntMuJoCoEnv
from torch import Tensor

from src.env.broken_env import BrokenEnv
from src.utils.common import label_frame

class BrokenAntBulletEnv(AntMuJoCoEnv, BrokenEnv):
  """
  """
  def __init__(self) -> None:
    AntMuJoCoEnv.__init__(self)
    BrokenEnv.__init__(self, 8, [1.]*8)

    ## change the state to exclude `crfc` of the original env
    high = np.inf * np.ones(27)
    self.observation_space = gym.spaces.Box(-high, high)
  
  """
  """
  @property
  def goal(self):
    ## same as None but can't use None since None can't be converted into tensor
    ## nevertheless, reward computation ignores this value
    return 0.
  
  """
  Needed to compute reward for MPC.
  Some parameters may not be used since this is intended to unify the API only.
  """
  @staticmethod
  def reward_func(obs: Tensor, act: Tensor, next_obs: Tensor, goal: Tensor) -> Tensor:
    return next_obs[..., 13]

  """
  """
  def render(self, mode: str, labels: Dict):
    frame = AntMuJoCoEnv._render(self, mode)
    return label_frame(frame, **labels)
  
  """
  """
  def reset(self, actuator_damage: List[float] = None):
    obs = AntMuJoCoEnv._reset(self)

    if actuator_damage:
      BrokenEnv.set_actuator_damage(self, actuator_damage)
    
    return obs[:27]

  """
  """
  def step(self, a):
    obs, rew, done, info = AntMuJoCoEnv.step(self,
      BrokenEnv.apply_damage(self, a))
    
    return obs[:27], rew, done, info