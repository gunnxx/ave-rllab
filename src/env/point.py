import numpy as np
from torch import Tensor
from torch.linalg import norm
from gym import Env
from gym.spaces import Box

class PointEnv(Env):
  """
  Simple point goal-reaching environment.
  This is very useful for debuging.
  Credits: https://github.com/cbfinn/maml_rl
  """
  def __init__(self, goal = None):
    self.reset(goal)
    self.action_space = Box(low=-0.1, high=0.1, shape=(2,))
    self.observation_space = Box(low=-np.inf, high=np.inf, shape=(2,))
  
  """
  """
  @property
  def goal(self):
    return self._goal
  
  """
  Needed to compute reward for MPC.
  Some parameters may not be used since this is intended to unify the API only.
  """
  @staticmethod
  def reward_func(obs: Tensor, act: Tensor, next_obs: Tensor, goal: Tensor) -> Tensor:
    return -norm(goal - next_obs, dim=-1)

  """
  """
  def reset(self, goal=None):
    if goal is not None:
      self._goal = np.array(goal, dtype=np.float64)
    else:
      self._goal = np.random.uniform(-0.5, 0.5, size=(2,))
    
    self._state = np.array([0., 0.])
    return np.copy(self._state)
  
  """
  """
  def step(self, action):
    if type(action) == Tensor: action = action.cpu().numpy()
    self._state = self._state + action

    reward = -np.linalg.norm(self._state - self._goal).item()
    done = abs(reward) < 0.01

    return self._state, reward, done, {'goal': self._goal}
  
  """
  Render square of size (-0.7, 0.7) in x and y axis.
  Discretization level = 0.05.
  `mode` and `labels` are needed like other envs.
  They are unused parameters in this environment.
  """
  def render(self, mode, labels):
    img = np.ones((29, 29)).astype('uint8') * 255
    
    g = np.floor(self._goal*20).astype('int') + (14, 14)
    s = np.floor(self._state*20).astype('int') + (14, 14)

    img[g[0], g[1]] = 0
    try:
      img[s[0], s[1]] = 100
    except:
      print("State=(%.2f, %.2f) is out of rendering bound."
      % tuple(self._state))
    
    return img