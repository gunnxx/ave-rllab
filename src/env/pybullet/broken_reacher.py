from typing import Dict, List

import gym.spaces
import numpy as np
from torch import Tensor, exp, norm, pow
from pybulletgym.envs.roboschool.envs.manipulation.reacher_env import ReacherBulletEnv

from src.env.broken_env import BrokenEnv
from src.utils.common import label_frame

class BrokenReacherBulletEnv(ReacherBulletEnv, BrokenEnv):
  """
  """
  def __init__(self) -> None:
    ReacherBulletEnv.__init__(self)
    BrokenEnv.__init__(self, 2, [1.]*2)

    # for better rendering result
    self._cam_pitch = -90
    self._cam_dist  = 0.6
  
    # change the state from 9 to 6
    high = np.inf * np.ones([6])
    self.observation_space = gym.spaces.Box(-high, high)
  
  """
  """
  @property
  def goal(self):
    target_x, _ = self.robot.jdict["target_x"].current_position()
    target_y, _ = self.robot.jdict["target_y"].current_position()
    return np.array([target_x, target_y])
  
  """
  Needed to compute reward from predicted_obs from model.
  """
  @staticmethod
  def reward_from_obs_and_goal(obs: Tensor, goal: Tensor) -> Tensor:
    sse = pow(obs[..., -2:] - goal, 2).sum(-1)
    spd = norm(obs[..., [1, 3]], dim=-1)
    return exp(-sse) * 100 - spd

  """
  """
  def render(self, mode: str, labels: Dict):
    frame = ReacherBulletEnv._render(self, mode)
    return label_frame(frame, **labels)

  """
  """
  def reset(self, actuator_damage: List[float] = None):
    obs = ReacherBulletEnv._reset(self)

    if actuator_damage:
      BrokenEnv.set_actuator_damage(self, actuator_damage)

    t = self.robot.central_joint.current_relative_position()
    fingertip = self.robot.fingertip.pose().xyz()

    reduced_obs = np.array([
      t[0],         # theta = central_joint
      t[1],         # theta_dot
      obs[7],       # gamma = elbow_joint
      obs[8],       # gamma_dot
      fingertip[0], # end-effector_x
      fingertip[1]  # end-effector_y
    ])

    return reduced_obs

  """
  """
  def step(self, a):
    obs, rew, done, info = ReacherBulletEnv.step(self,
      BrokenEnv.apply_damage(self, a))
    
    t = self.robot.central_joint.current_relative_position()
    fingertip = self.robot.fingertip.pose().xyz()
    
    reduced_obs = np.array([
      t[0],         # theta = central_joint
      t[1],         # theta_dot
      obs[7],       # gamma = elbow_joint
      obs[8],       # gamma_dot
      fingertip[0], # end-effector_x
      fingertip[1]  # end-effector_y
    ])
    
    return reduced_obs, rew, done, info
