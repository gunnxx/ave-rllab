from typing import Callable, Dict, Type

from gym import Env
from torch.optim import Optimizer

from buffer.buffer import Buffer
from buffer.consecutive_buffer import ConsecutiveBuffer
from model.model import Model
from model.stochastic_model import StochasticModel
from utils.common import check_config_keys
from utils.logger import Logger

class AdaptiveLearner:
  # List of constructor parameters
  # This will be used to check config in run.py
  REQUIRED_CONFIG_KEYS = [
    "env",
    "logger",
    "batch_size",
    "training_epoch",
    "num_collection_steps",
    "max_episode_length",
    "model_dynamics_type",
    "model_dynamics_params",
    "buffer_type",
    "buffer_params",
    "num_past_obs",
    "num_future_obs",
    "task_sampling_freq",
    "planning_horizon"
  ]

  # Part of REQUIRED_CONFIG_KEYS to instantiate model
  MODEL_CONFIG_KEYS = {
    "model_dynamics_type": "model_dynamics_params"
  }

  # Part of REQUIRED_CONFIG_KEYS to instantiate buffer
  BUFFER_CONFIG_KEYS = {
    "buffer_type": "buffer_params"
  }

  # Part of REQUIRED_CONFIG_KEYS to instantiate optimizer
  OPTIMIZER_CONFIG_KEYS = {
    "meta_optimizer_type": "meta_optimizer_params"
  }

  # Part of REQUIRED_CONFIG_KEYS to instantiate controller
  CONTROLLER_CONFIG_KEYS = {
    "mpc_controller": None
  }

  def __init__(self,
    env: Env,
    logger: Logger,
    batch_size: int,
    task_learning_rate: float,
    training_epoch: int,
    num_collection_steps: int,
    testing_episode: int,
    max_episode_length: int,
    policy_start_steps: int,
    model_dynamics_type: Type[StochasticModel],
    model_dynamics_params: Dict,
    buffer_type: Type[ConsecutiveBuffer],
    buffer_params: Dict,
    meta_optimizer_type: Type[Optimizer],
    meta_optimizer_params: Dict,
    mpc_controller: Callable,
    num_inner_grad_steps: int,
    num_past_obs: int,
    num_future_obs: int,
    task_sampling_freq: int,
    planning_horizon: int) -> None:

    self.env = env
    self.logger = logger

    # general training optimization hyperparams
    self.batch_size : int = batch_size
    self.task_learning_rate : float = task_learning_rate

    # general training loop hyperparams
    self.training_epoch = training_epoch
    self.num_collection_steps = num_collection_steps
    self.testing_episode = testing_episode
    self.max_episode_length = max_episode_length
    self.policy_start_steps = policy_start_steps

    # meta-learning and adaptation hyperparams
    self.num_inner_grad_steps = num_inner_grad_steps
    self.num_past_obs = num_past_obs
    self.num_future_obs = num_future_obs
    self.task_sampling_freq = task_sampling_freq
    self.planning_horizon = planning_horizon

    # model-based predictive controller
    self.controller = mpc_controller

    # instantiate learnable model dynamics
    self.model_dynamics: StochasticModel = Model.instantiate_model(
      model_dynamics_type,
      model_dynamics_params)

    # instantiate replay buffer
    self.buffer: Buffer = Buffer.instantiate_buffer(
      buffer_type,
      buffer_params)
    
    # instantiate optimizer for meta-update
    self.meta_optimizer = meta_optimizer_type(
      self.model_dynamics.parameters(),
      **meta_optimizer_params)
  
  @staticmethod
  def validate_config(config: Dict) -> None:
    check_config_keys(config, AdaptiveLearner.REQUIRED_CONFIG_KEYS)
    
    assert config["planning_horizon"] < config["num_future_obs"], "Planning \
      horizon should be less than adaptation horizon to better capture local \
      context."

    assert config["num_collection_steps"] >= config["num_past_obs"] + \
      config["num_future_obs"], "Collection steps should be greater than M+k \
      as in the paper"

    # param to buffer that set the contagious-level should be equals
    # to num_past_obs + num_future_obs
  
  def _model_meta_learn(self):
    raise NotImplementedError()
  
  def _model_online_adaptation(self):
    raise NotImplementedError()

  def run(self) -> None:
    t = 0
    for epoch in range(self.training_epoch):      
      ##########################
      ## COLLECT TRAJECTORIES ##
      ##########################
      if epoch % self.task_sampling_freq == 0:
        obs = self.env.reset()
        episode_ret = 0
        episode_len = 0

        for _ in range(self.num_collection_steps):
          # start with random policy,
          # model-based planning afterwards
          if t < self.policy_start_steps:
            act = self.env.action_space.sample()
          else:
            self._model_online_adaptation()
            act = self.controller()
          
          # step in the environment
          next_obs, rew, done, info = self.env.step(act)
          success = info.get('is_success', False)

          # store to replay buffer
          self.buffer.store(
            obs=obs,
            act=act,
            next_obs=next_obs,
            rew=rew)
          
          # update values
          obs = next_obs
          episode_ret += rew
          episode_len += 1
          t += 1

          # handle end of trajectory
          if done or success or episode_len == self.max_episode_length:
            obs = self.env.reset()
            episode_ret = 0
            episode_len = 0

      ################
      ## META-LEARN ##
      ################
      self._model_meta_learn()