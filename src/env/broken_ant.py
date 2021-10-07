from typing import Dict
from torch import Tensor
from pybulletgym.envs.roboschool.envs.locomotion.ant_env import AntBulletEnv

from src.env.broken_env import BrokenEnv
from src.utils.common import label_frame

class BrokenAntBulletEnv(AntBulletEnv, BrokenEnv):
  """
  """
  def __init__(self) -> None:
    AntBulletEnv.__init__(self)
    BrokenEnv.__init__(self, 8, [False]*8, [1.]*8)

  """
  """
  @staticmethod
  def reward_from_obs(obs: Tensor) -> Tensor:
    pass

  """
  """
  def render(self, mode: str, labels: Dict):
    frame = AntBulletEnv.render(self, mode)
    return label_frame(frame, **labels)

  """
  """
  def step(self, a):
    return AntBulletEnv.step(self,
      BrokenEnv.apply_damage(self, a))