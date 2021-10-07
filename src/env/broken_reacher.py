from typing import Dict
from torch import Tensor
from pybulletgym.envs.roboschool.envs.manipulation.reacher_env import ReacherBulletEnv

from src.env.broken_env import BrokenEnv
from src.utils.common import label_frame

class BrokenReacherBulletEnv(ReacherBulletEnv, BrokenEnv):
  """
  """
  def __init__(self) -> None:
    ReacherBulletEnv.__init__(self)
    BrokenEnv.__init__(self, 2, [False]*2, [1.]*2)

    # for better rendering result
    self._cam_pitch = -90
    self._cam_dist  = 0.6
  
  """
  """
  @staticmethod
  def reward_from_obs(obs: Tensor) -> Tensor:
    pass

  """
  """
  def render(self, mode: str, labels: Dict):
    frame = ReacherBulletEnv.render(self, mode)
    return label_frame(frame, **labels)

  """
  """
  def step(self, a):
    return ReacherBulletEnv.step(self,
      BrokenEnv.apply_damage(self, a))