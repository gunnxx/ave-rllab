from pybulletgym.envs.roboschool.envs.locomotion.ant_env import AntBulletEnv
from src.env.broken_env import BrokenEnv

class BrokenAntBulletEnv(AntBulletEnv, BrokenEnv):
  """
  """
  def __init__(self) -> None:
    AntBulletEnv.__init__(self)
    BrokenEnv.__init__(self, 8, [False]*8, [1.]*8)

  """
  """
  def step(self, a):
    return AntBulletEnv.step(self,
      BrokenEnv.apply_damage(self, a))