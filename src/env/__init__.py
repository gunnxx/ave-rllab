from gym.envs.registration import register

register(
  id='BrokenReacherPyBulletEnv-v0',
  entry_point='src.env.broken_reacher:BrokenReacherBulletEnv',
  max_episode_steps=150,
  reward_threshold=18.0,
  )

register(
  id='BrokenAntPyBulletEnv-v0',
  entry_point='src.env.broken_ant:BrokenAntBulletEnv',
  max_episode_steps=1000,
  reward_threshold=2500.0
  )