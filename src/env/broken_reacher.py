from pybulletgym.envs.roboschool.envs.manipulation.reacher_env import ReacherBulletEnv
from src.env.broken_env import BrokenEnv

class BrokenReacherBulletEnv(ReacherBulletEnv, BrokenEnv):
  """
  """
  def __init__(self) -> None:
    ReacherBulletEnv.__init__(self)
    BrokenEnv.__init__(self, 2, [False]*2, [1.]*2)

  """
  """
  def step(self, a):
    return ReacherBulletEnv.step(self,
      BrokenEnv.apply_damage(self, a))