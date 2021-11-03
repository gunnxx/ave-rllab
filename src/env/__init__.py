from gym.envs.registration import register

# --------------- basic env for testing ---------------

register(
  id='PointEnv-v0',
  entry_point='src.env.point:PointEnv',
  max_episode_steps=20,
  reward_threshold=0.
)

# --------------- inherits `pybulletgym` using `pybullet` engine ---------------

register(
  id='BrokenReacherPyBulletEnv-v0',
  entry_point='src.env.pybullet.broken_reacher:BrokenReacherBulletEnv',
  max_episode_steps=150,
  reward_threshold=18.0,
)

register(
  id='BrokenAntPyBulletEnv-v0',
  entry_point='src.env.pybullet.broken_ant:BrokenAntBulletEnv',
  max_episode_steps=1000,
  reward_threshold=2500.0
)

register(
  id='BrokenHalfCheetahPyBulletEnv-v0',
  entry_point='src.env.pybullet.broken_half_cheetah:BrokenHalfCheetahMujocoEnv',
  max_episode_steps=1000,
  reward_threshold=2500.0
)