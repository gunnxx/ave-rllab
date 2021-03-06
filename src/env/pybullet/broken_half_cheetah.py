from typing import Dict, List

import gym.spaces
import numpy as np
from pybulletgym.envs.mujoco.envs.locomotion.half_cheetah_env import HalfCheetahMuJoCoEnv
from torch import Tensor

from src.env.broken_env import BrokenEnv
from src.utils.common import label_frame

class BrokenHalfCheetahMujocoEnv(HalfCheetahMuJoCoEnv, BrokenEnv):  
  """
  """
  def __init__(self) -> None:
    HalfCheetahMuJoCoEnv.__init__(self)
    BrokenEnv.__init__(self, 6, [1.]*6)

    ## change the state from 17 to 18 to include x_torso
    high = np.inf * np.ones([18])
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
    ## 30 is just for scaling, no physical meaning
    return (next_obs[..., 0] - obs[..., 0]) * 30

  """
  """
  def render(self, mode: str, labels: Dict):
    frame = HalfCheetahMuJoCoEnv._render(self, mode)
    return label_frame(frame, **labels)
  
  """
  """
  def reset(self, actuator_damage: List[float] = None):
    _ = HalfCheetahMuJoCoEnv._reset(self)

    if actuator_damage:
      BrokenEnv.set_actuator_damage(self, actuator_damage)
    
    qpos = np.array([j.get_position() for j in self.robot.ordered_joints], dtype=np.float32).flatten()
    qvel = np.array([j.get_velocity() for j in self.robot.ordered_joints], dtype=np.float32).flatten()
    
    return np.concatenate([qpos, qvel])
  
  """
  """
  def step(self, a):
    _, rew, done, info = HalfCheetahMuJoCoEnv.step(self,
      BrokenEnv.apply_damage(self, a))
    
    qpos = np.array([j.get_position() for j in self.robot.ordered_joints], dtype=np.float32).flatten()
    qvel = np.array([j.get_velocity() for j in self.robot.ordered_joints], dtype=np.float32).flatten()
    obs = np.concatenate([qpos, qvel])

    return obs, rew, done, info