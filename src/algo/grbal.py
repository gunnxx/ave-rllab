from typing import Dict, Type

import gym
import numpy as np
import torch
import torch.optim as optim

from learn2learn import clone_module, update_module

from src.algo.algo import Algo, REGISTERED_OPTIM
from src.buffer.consecutive_buffer import ConsecutiveBuffer
from src.model.stochastic_model import StochasticModel
from src.utils.logger import Logger
from src.utils.common import cast_to_torch

class GrBAL(Algo):
  # List of constructor parameters
  # This will be used to check config in run.py
  REQUIRED_CONFIG_KEYS = {
    "env": None,
    "logger": None,
    "batch_size": 8,
    "task_learning_rate": 1e-3,
    "training_epoch": 32,
    "num_collection_steps": 1000,
    "testing_episode": None,
    "max_episode_length": 1000,
    "policy_start_steps": 1000,
    "model_dynamics_type": "gaussian_mlp_model",
    "model_dynamics_params": {},
    "buffer_type": "consecutive_buffer",
    "buffer_params": {},
    "meta_optimizer_type": "adam",
    "meta_optimizer_params": {},
    "num_inner_grad_steps": 1,
    "num_past_obs": 15,
    "num_future_obs": 5,
    "task_sampling_freq": 1,
    "planning_horizon": 4,
    "n_trajectory": 256,
    "is_first_order": True,
    "is_nested_grad": False
  }

  # Part of REQUIRED_CONFIG_KEYS to instantiate model
  MODEL_CONFIG_KEYS = {
    "model_dynamics_type": "model_dynamics_params"
  }

  # Part of REQUIRED_CONFIG_KEYS to instantiate buffer
  BUFFER_CONFIG_KEYS = {
    "buffer_type": "buffer_params"
  }

  """
  params:
    env:
      environment object used
    logger:
      logger object
    batch_size:
      number of tasks used for MAML
    task_learning_rate:
      learning rate for vanilla SGD on each task
    training_epoch:
      number of training epochs (1-epoch = 1-meta-update)
    num_collection_steps:
      number of steps on environment to fill buffer
    testing_episode:
      number of testing episode
    max_episode_length:
      maximum episode length
    policy_start_steps:
      starting step to use model+MPC to select action
      (before this using random actions sampled from
      `env.action_space`)
    model_dynamics_type:
      type of model dynamics used to represent the real
      dynamics, must be derived from `StochasticModel`
    model_dynamics_params:
      parameters to instantiate the `model_dynamics_type`
    buffer_type:
      type of buffer used to store transition dynamics,
      only accept `ConsecutiveBuffer`
    buffer_params:
      parameters to instantiate `buffer_type`
    meta_optimizer_type:
      type of optimizer used to optimize the meta-model,
      must be derived from `torch.optim.Optimizer`
    meta_optimizer_params:
      parameters to isntantiate `meta_optimizer_type`
    num_inner_grad_steps:
      number of vanilla SGD steps to take on each tasks
    num_past_obs:
      number of samples on each tasks from the past
      used to adapt the task-model i.e. `M` in the paper
    num_future_obs:
      number of samples on each tasks from the future
      used to adapt the meta-model i.e. `K` in th paper
    task_sampling_freq:
      how often to collect trajectories i.e.
      `epoch % task_sampling_freq == 0`
    planning_horizon:
      horizon used for MPC i.e. random-shooting
    n_trajectory:
      number of trajectories sampled for random-shooting
    is_first_order:
      whether to use first-order MAML
    is_nested_grad:
      whether inner gradient is computed w.r.t meta-param
  
  notes:
  `is_first_order = True` means using first-order MAML,
  regardless the value of nested grad. `is_nested_grad = True`
  means using nested-MAML and `is_nested_grad = False` means
  using original-MAML. If `num_inner_grad_steps = 1`, then
  nested-MAML is the same as original-MAML.
  """
  def __init__(self,
    env: gym.Env,
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
    meta_optimizer_type: str,
    meta_optimizer_params: Dict,
    num_inner_grad_steps: int,
    num_past_obs: int,
    num_future_obs: int,
    task_sampling_freq: int,
    planning_horizon: int,
    n_trajectory: int,
    is_first_order: bool,
    is_nested_grad: bool) -> None:

    self.env : gym.Env = env
    self.logger : Logger = logger

    # general training optimization hyperparams
    self.batch_size : int = batch_size
    self.task_learning_rate : float = task_learning_rate

    # general training loop hyperparams
    self.training_epoch : int = training_epoch
    self.num_collection_steps : int = num_collection_steps
    self.testing_episode : int = testing_episode
    self.max_episode_length : int = max_episode_length
    self.policy_start_steps : int = policy_start_steps

    # meta-learning and adaptation hyperparams
    self.num_inner_grad_steps : int = num_inner_grad_steps
    self.num_past_obs : int = num_past_obs
    self.num_future_obs : int = num_future_obs
    self.task_sampling_freq : int = task_sampling_freq
    self.planning_horizon : int = planning_horizon
    self.n_trajectory : int = n_trajectory

    # gradient spec
    self.is_first_order : bool = is_first_order
    self.is_nested_grad : bool = is_nested_grad

    # instantiate learnable model dynamics
    self.model_dynamics: StochasticModel = \
      model_dynamics_type.instantiate_model(
      model_dynamics_params)

    # instantiate replay buffer
    self.buffer: ConsecutiveBuffer = \
      buffer_type.instantiate_buffer(
      buffer_params)
    
    # instantiate optimizer for meta-update
    optim_type = REGISTERED_OPTIM[meta_optimizer_type]
    self.meta_optimizer : optim.Optimizer = optim_type(
      self.model_dynamics.parameters(),
      **meta_optimizer_params)


  """
  """
  @staticmethod
  def validate_params(params: Dict) -> None:
    assert params["planning_horizon"] < params["num_future_obs"], \
    """Planning horizon should be less than adaptation horizon to
    better capture local context."""

    assert params["num_collection_steps"] >= params["num_past_obs"] \
    + params["num_future_obs"], """Collection steps should be greater
    than the consecutive_size of the buffer. It needs to at least
    store one entry in the buffer."""
    
    assert params["num_past_obs"] + params["num_future_obs"] == \
    params["buffer_params"]["consecutive_size"], """This should match
    because this chunk is considered as one situation."""
    
    assert params["env"].action_space.shape[0] + \
    params["env"].observation_space.shape[0] == \
    params["model_dynamics_params"]["input_dim"], """Input to the
    network is action and state tuple."""
    
    assert params["env"].observation_space.shape[0] == \
    params["model_dynamics_params"]["output_dim"], """Output of the
    network is the next state."""
  
  
  """
  """
  def _model_online_adaptation(self) -> StochasticModel:
    # take the most recent entry in the buffer for model adaptiation
    samples = self.buffer.sample_last_n(1)

    obs = samples["obs"][0][-self.num_past_obs:, :]
    act = samples["act"][0][-self.num_past_obs:, :]
    next_obs = samples["next_obs"][0][-self.num_past_obs:, :]

    # zero-ing out gradients
    self.model_dynamics.zero_grad()

    # clone model because we are not doing in-place update
    clone_model_dynamics = clone_module(self.model_dynamics)

    # adaptation
    for _ in range(self.num_inner_grad_steps):
      log_prob_model = clone_model_dynamics.log_prob_from_data(
        torch.cat([obs, act], dim=-1), next_obs)

      # whether nested-MAML, orig-MAML, or fo-MAML
      if self.is_nested_grad:
        grad = torch.autograd.grad(
          -log_prob_model.sum(),
          self.model_dynamics.parameters(),
          create_graph=(not self.is_first_order))
      else:
        grad = torch.autograd.grad(
          -log_prob_model.sum(),
          clone_model_dynamics.parameters(),
          create_graph=(not self.is_first_order))
      
      clone_model_dynamics = update_module(
        clone_model_dynamics,
        updates=tuple(self.task_learning_rate*g for g in grad))

    return clone_model_dynamics

  
  """
  """
  def _model_meta_learn(self) -> None:
    # batch_size is the number of tasks
    # samples = (batch_sz, num_past_obs + num_future_obs, feature_sz)
    samples = self.buffer.sample_batch(self.batch_size)
    
    obs = samples["obs"]
    obs_support = obs[:, :self.num_past_obs, :]
    obs_query   = obs[:, self.num_past_obs:, :]

    act = samples["act"]
    act_support = act[:, :self.num_past_obs, :]
    act_query   = act[:, self.num_past_obs:, :]

    next_obs = samples["next_obs"]
    next_obs_support = next_obs[:, :self.num_past_obs, :]
    next_obs_query   = next_obs[:, self.num_past_obs:, :]

    # zero-ing out gradients
    self.model_dynamics.zero_grad()

    # training
    meta_grads = tuple(torch.zeros_like(p)
      for p in self.model_dynamics.parameters())

    for i in range(self.batch_size):
      clone_model_dynamics = clone_module(self.model_dynamics)

      # inner grad-descent loop (task training)
      for _ in range(self.num_inner_grad_steps):
        log_prob_model = clone_model_dynamics.log_prob_from_data(
          torch.cat([obs_support[i], act_support[i]], dim=-1),
          next_obs_support[i])
        
        # whether nested-MAML, orig-MAML, or fo-MAML
        if self.is_nested_grad:
          grad = torch.autograd.grad(
            -log_prob_model.sum(),
            self.model_dynamics.parameters(),
            create_graph=(not self.is_first_order))
        else:
          grad = torch.autograd.grad(
            -log_prob_model.sum(),
            clone_model_dynamics.parameters(),
            create_graph=(not self.is_first_order))
        
        clone_model_dynamics = update_module(
          clone_model_dynamics,
          updates=tuple(self.task_learning_rate*g for g in grad))
      
      # meta-gradient
      log_prob_model = clone_model_dynamics.log_prob_from_data(
        torch.cat([obs_query[i], act_query[i]], dim=-1),
        next_obs_query[i])
      
      grad = torch.autograd.grad(
        -log_prob_model.sum(),
        self.model_dynamics.parameters())
      
      meta_grads = tuple(mg + g for mg, g in zip(meta_grads, grad))
    
    # meta-update
    for p, mg in zip(self.model_dynamics.parameters(), meta_grads):
      p.grad = mg/self.batch_size
    self.meta_optimizer.step()
  
  
  """
  """
  def _random_shooting_controller(self,
    model: StochasticModel,
    n_trajectory: int,
    curr_obs: np.ndarray) -> np.ndarray:
    # initialize reward
    rewards = torch.zeros(n_trajectory)

    # sample actions uniformly
    act_dim  = self.env.action_space.shape[0]
    act_low  = self.env.action_space.low[0]
    act_high = self.env.action_space.high[0]
    act = torch.rand(self.planning_horizon, n_trajectory, act_dim)
    act = (act_low - act_high) * act + act_high

    # from (obs_dim) to (n_trajectory, obs_dim)
    curr_obs = cast_to_torch(curr_obs, dtype=torch.float32)
    curr_obs = curr_obs.unsqueeze(dim=0).repeat(n_trajectory, 1)

    '''
    TODO:
    Abstract reward computation!
    We can move the reward computation to @staticmethod
    of the environment whose parameter is current_obs.
    '''
    # to calculate reward
    ## reward = new_potential - old_potential
    ## potential = -L2 distance to target
    if self.env.spec.id == 'BrokenReacherPyBulletEnv-v0':
      old_potential = -np.linalg.norm(curr_obs[:, 2:3], axis=1)

    ## reward = speed towards x-axis (vx_body * cos angle_to_target)
    ## potential = 0, recording potential just to unify the API later
    elif self.env.spec.id == 'BrokenAntPyBulletEnv-v0':
      old_potential = 0

    else:
        raise NotImplementedError("Reward func is hard-coded, \
          choose only the listed environments on src/env")

    # rollout the sampled action
    for i in range(self.planning_horizon):
      with torch.no_grad():
        next_obs, _ = model(
          torch.cat([curr_obs, act[i]], dim=-1),
          deterministic=False,
          with_logprob=False)
      
      '''
      TODO:
      Abstract reward computation!
      We can move the reward computation to @staticmethod
      of the environment whose parameter is current_obs.
      '''
      if self.env.spec.id == 'BrokenReacherPyBulletEnv-v0':
        potential = -np.linalg.norm(next_obs[:, 2:3], axis=1)
        reward = potential - old_potential
        old_potential = potential
      
      elif self.env.spec.id == 'BrokenAntPyBulletEnv-v0':
        potential = next_obs[:, 2] * next_obs[:, 3]
        reward = potential - old_potential
        old_potential = 0
      
      rewards += reward
      curr_obs = next_obs
    
    # choose the first action of the best trajectory
    best_idx = torch.argmax(rewards)
    return act[0, best_idx, :].numpy()

  
  """
  """
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
            adapted_model = self._model_online_adaptation()
            act = self._random_shooting_controller(
              adapted_model, self.n_trajectory, obs)
          
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