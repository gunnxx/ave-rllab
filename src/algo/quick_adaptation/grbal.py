from typing import Any, Dict, List, Tuple, Type

import numpy as np
import torch
import torch.nn as  nn

from copy import deepcopy
from gym import Env
from learn2learn import clone_module, update_module
from torch import Tensor
from torch.autograd import grad
from torch.optim import Adam
from tqdm import tqdm

from src.algo.quick_adaptation.base import QuickAdaptBase
from src.buffer.buffer import Buffer
from src.model.deterministic_model import DeterministicModel
from src.utils.common import cast_to_torch, warn_and_ask
from src.utils.metrics import EarlyStopMetric

class GrBAL(QuickAdaptBase):
  # List of constructor parameters along its default values.
  # This will be used to construct default config in run.py.
  REQUIRED_CONFIG_KEYS = {
    "env": None,
    "device": "cpu",
    "seed": 1501,
    "exp_dir": "exp/grbal",
    "num_iter": 50,
    "num_random_iter": 1,
    "max_epoch_per_iter": 5000,
    "render_freq": 1,
    "batch_size": 512,
    "num_episode_per_task": 100,
    "max_episode_length": 100,
    "mpc_horizon": 16,
    "mpc_num_trajectories": 4096,
    "buffer_type": "random_buffer",
    "buffer_params": {},
    "model_type": "deterministic_mlp_model",
    "model_params": {},
    "training_damages": [[1., 1.]],
    "validation_damages": [[1., 1.]],
    "testing_damages": [[1., 1.]],
    "task_learning_rate": 1e-4,
    "meta_learning_rate": 1e-2,
    "num_inner_grad_steps": 10,
    "num_past_obs": 32,
    "num_future_obs": 32,
    "is_first_order": True,
    "is_nested_grad": False
  }

  # The keys and values of MODEL_CONFIG_KEYS point to keys in REQUIRED_CONFIG_KEYS.
  # These will be used to change `model_type` from str to class variable of the model.
  # `model_params` will be overloaded by default params from the `model_type` class.
  MODEL_CONFIG_KEYS = {
    "model_type": "model_params"
  }

  # The keys and values of BUFFER_CONFIG_KEYS point to keys in REQUIRED_CONFIG_KEYS.
  # These will be used to change `buffer_type` from str to class variable of the model.
  # `buffer_params` will be overloaded by default params from the `buffer_type` class.
  BUFFER_CONFIG_KEYS = {
    "buffer_type": "buffer_params"
  }

  """
  params:
  """
  def __init__(self,
    env: Env,
    device: str,
    seed: int,
    exp_dir: str,
    num_iter: int,
    num_random_iter: int,
    max_epoch_per_iter: int,
    render_freq: int,
    batch_size: int,
    num_episode_per_task: int,
    max_episode_length: int,
    mpc_horizon: int,
    mpc_num_trajectories: int,
    buffer_type: Type[Buffer],
    buffer_params: Dict[str, Any],
    model_type: Type[DeterministicModel],
    model_params: Dict[str, Any],
    training_damages: List[List[float]],
    validation_damages: List[List[float]],
    testing_damages: List[List[float]],
    task_learning_rate: float,
    meta_learning_rate: float,  
    num_inner_grad_steps: int,
    num_past_obs: int,
    num_future_obs: int,
    is_first_order: bool,
    is_nested_grad: bool) -> None:

    ## based on original GrBAL implementation
    ## https://github.com/iclavera/learning_to_adapt
    is_store_per_episode = True

    super().__init__(env, device, seed, exp_dir, num_iter,
    num_random_iter, max_epoch_per_iter, render_freq, 
    batch_size, num_episode_per_task, max_episode_length, 
    num_past_obs, mpc_horizon, is_store_per_episode, buffer_type, 
    buffer_params, training_damages, validation_damages, 
    testing_damages)

    ## training hyperparams
    self.task_learning_rate : int = task_learning_rate
    self.meta_learning_rate : int = meta_learning_rate
    self.num_inner_grad_steps : int = num_inner_grad_steps
    self.num_future_obs : int = num_future_obs
    self.loss_fn : nn.MSELoss = nn.MSELoss()

    ## gradient spec
    self.is_first_order : bool = is_first_order
    self.is_nested_grad : bool = is_nested_grad

    ## MPC hyperparams
    self.mpc_num_trajectories : int = mpc_num_trajectories

    ## initialize model
    self.model : DeterministicModel = model_type.instantiate_model(model_params)
    self.model.to(self.device)

    ## optimizer
    self.optim : Adam = Adam(self.model.parameters(), meta_learning_rate)

    ## store (mean, std) of each dimension in the buffer
    self.normalizer : Dict[Tuple[Tensor]] = None
  
  """
  """
  def adapt_model(self, buffer: Dict[str, List[np.ndarray]]) -> Any:
    ## no adaptation
    if buffer["size"] == 0:
      return clone_module(self.model)

    clone_model = clone_module(self.model)
    clone_model.train()

    ## prepare the input
    buffer = {k: cast_to_torch(v, torch.float32, self.device) for k, v in buffer.items()}
    buffer = self._process_sample(buffer)
    in_ = torch.cat([buffer["obs"], buffer["act"]], dim=-1)

    ## adaptation
    for _ in range(self.num_inner_grad_steps):
      clone_model.zero_grad()

      ## forward and compute loss
      delta_pred = clone_model(in_)
      loss = self.loss_fn(delta_pred, buffer["delta"])

      ## nested-MAML / orig-MAML / fo-MAML
      ## see description at the top
      if self.is_nested_grad:
        g = grad(loss, self.model.parameters(), create_graph=(not self.is_first_order))
      else:
        g = grad(loss, clone_model.parameters(), create_graph=(not self.is_first_order))
      
      ## update using the computed gradient
      updates = [-self.task_learning_rate * g_ for g_ in g]
      clone_model = update_module(clone_model, updates=updates)
    
    return clone_model
  
  """
  """
  def mpc(self, model: Any, obs: np.ndarray) -> np.ndarray:
    model.eval()

    ## reward container of each samples
    returns = torch.zeros(self.mpc_num_trajectories, device=self.device)

    ## sample actions uniformly
    act_dim = self.env.action_space.shape[0]
    act_low = self.env.action_space.low[0]
    act_high = self.env.action_space.high[0]

    shape = (self.mpc_horizon, self.mpc_num_trajectories, act_dim)
    act = torch.rand(*shape, device=self.device)
    act = (act_low - act_high) * act + act_high

    ## (obs_dim) -> (mpc_num_trajectories, obs_dim)
    obs = cast_to_torch(obs, torch.float32, self.device)
    obs = obs.unsqueeze(dim=0).repeat(self.mpc_num_trajectories, 1)

    ## normalized all actions
    act = (act - self.normalizer["act"][0]) / (self.normalizer["act"][1] + 1e-10)

    ## rollout sampled actions
    with torch.no_grad():
      for i in range(self.mpc_horizon):
        ## normalized copy of current observation
        obs_ = deepcopy(obs)
        obs_ = (obs_ - self.normalizer["obs"][0]) / (self.normalizer["obs"][1] + 1e-10)

        ## predict normalized-data
        in_ = torch.cat([obs_, act[i]], dim=-1)
        diff = model(in_)

        ## denormalize
        diff *= self.normalizer["delta"][1] + 1e-10
        diff += self.normalizer["delta"][0]

        ## unnormalized next_obs
        obs += diff

        ## compute reward: @staticmethod
        goal = torch.tensor(self.env.goal, device=self.device)
        reward = self.env.reward_from_obs_and_goal(obs, goal)
        returns += reward

    ## choose the first action of the best trajectory
    best_idx = torch.argmax(returns)
    return act[0, best_idx, :].cpu().numpy()
  
  """
  """
  def meta_learn_model(self, iter: int) -> None:
    early_stop_metric = EarlyStopMetric(patience=3, decay=0)

    ## computed values are stored in `self.normalizer`
    self._compute_normalization_constant()

    ## compute num of loops to be counted as one epoch
    total_train_data = self.num_training_task * self.buffer[0].size * self.max_episode_length
    total_valid_data = self.num_validation_task * self.buffer[0].size * self.max_episode_length
    
    num_steps_per_epoch_train = int(total_train_data / self.batch_size)
    num_steps_per_epoch_valid = int(total_valid_data / self.batch_size)

    ## training
    for epoch in tqdm(range(self.max_epoch_per_iter), "Training", position=0, leave=False):
      self.model.train()

      ## container for meta-loss and valid-loss
      task_losses = np.zeros(num_steps_per_epoch_train)
      meta_losses = np.zeros(num_steps_per_epoch_train)
      valid_losses = np.zeros(num_steps_per_epoch_valid)

      for step in range(num_steps_per_epoch_train):
        clone_model = clone_module(self.model)

        ## (batch_size, num_samples_to_adapt + num_future_obs, feature_sz)
        batch = self._get_batch(is_train=True)

        obs_sup = batch["obs"][:, :self.num_samples_to_adapt, :]
        obs_qry = batch["obs"][:, self.num_samples_to_adapt:, :]

        act_sup = batch["act"][:, :self.num_samples_to_adapt, :]
        act_qry = batch["act"][:, self.num_samples_to_adapt:, :]

        delta_sup = batch["delta"][:, :self.num_samples_to_adapt, :]
        delta_qry = batch["delta"][:, self.num_samples_to_adapt:, :]

        ## prepare the input
        in_sup = torch.cat([obs_sup, act_sup], dim=-1)
        in_qry = torch.cat([obs_qry, act_qry], dim=-1)

        ## inner SGD loop ~ task training
        for _ in range(self.num_inner_grad_steps):
          clone_model.zero_grad()

          ## forward and compute loss
          delta_pred = clone_model(in_sup)
          loss = self.loss_fn(delta_pred, delta_sup)

          ## nested-MAML / orig-MAML / fo-MAML
          ## see description at the top
          if self.is_nested_grad:
            g = grad(loss, self.model.parameters(), create_graph=(not self.is_first_order))
          else:
            g = grad(loss, clone_model.parameters(), create_graph=(not self.is_first_order))
          
          ## update using the computed gradient
          updates = [-self.task_learning_rate * g_ for g_ in g]
          clone_model = update_module(clone_model, updates=updates)
        
        ## meta-gradient computation
        delta_pred = clone_model(in_qry)
        meta_loss = self.loss_fn(delta_pred, delta_qry)
        mg = grad(meta_loss, self.model.parameters())

        ## overload `.grad` property of model parameters
        for p_, mg_ in zip(self.model.parameters(), mg):
          p_.grad = mg_
        
        ## one step of optimization ~ meta-optimization
        self.optim.step()
      
        ## save the loss
        task_losses[step] = loss.item()
        meta_losses[step] = meta_loss.item()
        self.logger.epoch_store(task_loss=loss.item(), meta_loss=loss.item())
      
      self.model.eval()

      ## validation
      for step in range(num_steps_per_epoch_valid):
        ## prepare the input
        batch = self._get_batch(is_train=False)
        in_ = torch.cat([batch["obs"], batch["act"]], dim=-1)

        ## compute validation loss
        with torch.no_grad():
          delta_pred = self.model(in_)
          loss = self.loss_fn(delta_pred, batch["delta"])
        
        ## save the loss
        valid_losses[step] = loss.item()
        self.logger.epoch_store(valid_loss=loss.item())
      
      avg_meta_loss = meta_losses.mean()
      avg_valid_loss = valid_losses.mean()

      ## log: self.logger and stdout
      log = {"iter": iter, "epoch": epoch, "meta-loss": avg_meta_loss, "valid-loss": avg_valid_loss}
      self.logger.store(training=log)
      self.logger.epoch_store(iter=iter, epoch=epoch)
      self.logger.dump()

      tqdm.write("Iter %02d Epoch %04d || %.04f || %.04f" %
        (iter, epoch, avg_meta_loss.item(), avg_valid_loss.item()))

      ## early-stop based on average validation loss
      early_stop_metric(avg_valid_loss)
      if early_stop_metric.is_stop():
        tqdm.write("Early stopping is called!")
        break 
  
  """
  """
  def handle_end_of_iteration(self, iter: int, time: float) -> None:
    chkpt = self._get_checkpoint()
    self.logger.torch_save(chkpt, "latest.pt")
  
  """
  """
  def _compute_normalization_constant(self) -> None:
    obs, act, delta = [], [], []
    for buffer in self.buffer:
      obs += [buffer.data['obs']]
      act += [buffer.data['act']]
      delta += [buffer.data['next_obs'] - buffer.data['obs']]
    
    ## (num_episode_per_task * num_tasks, max_episode_length, feature_sz)
    obs = torch.cat(obs)
    act = torch.cat(act)
    delta = torch.cat(delta)

    ## compute the normalizer
    self.normalizer = {
      'obs': (obs.mean(dim=[0, 1]), obs.std(dim=[0, 1])),
      'act': (act.mean(dim=[0, 1]), act.std(dim=[0, 1])),
      'delta': (delta.mean(dim=[0, 1]), delta.std(dim=[0, 1]))
    }
  
  """
  """
  def _process_sample(self, sample: Dict[str, Tensor]) -> Dict[str, Tensor]:
    obs_mean, obs_std = self.normalizer['obs']
    act_mean, act_std = self.normalizer['act']

    ## sample["next_obs"] is optional e.g. mpc()
    try:
      dlt_mean, dlt_std = self.normalizer['delta']
      delta = sample['next_obs'] - sample['obs']
      delta = (delta - dlt_mean) / (dlt_std + 1e-10)
    except:
      delta = None
    
    return {
      'obs': (sample['obs'] - obs_mean) / (obs_std + 1e-10),
      'act': (sample['act'] - act_mean) / (act_std + 1e-10),
      'delta': delta
    }
  
  """
  """
  def _get_batch(self, is_train: bool) -> Dict[str, Tensor]:
    ## start and end index of task index sampling
    if is_train:
      start_idx = 0
      end_idx = self.num_training_task
    else:
      start_idx = self.num_training_task
      end_idx = self.num_training_task + self.num_validation_task
    
    ## lowest and highest timestep index to ensure (backward) consecutive sampling
    low = 0 + self.num_samples_to_adapt + self.num_future_obs
    high = self.max_episode_length
    
    ## task and timestep index sampling
    task_idx = np.random.randint(start_idx, end_idx, self.batch_size)
    end_time_idx = np.random.randint(low, high, self.batch_size)
    start_time_idx = end_time_idx - self.num_samples_to_adapt - self.num_future_obs

    ## container for the data
    obs = []
    act = []
    next_obs = []

    ## get the data
    for idx, start, end in zip(task_idx, start_time_idx, end_time_idx):
      sample = self.buffer[idx].sample_batch(1)
      obs += [sample["obs"][:, start:end, :]]
      act += [sample["act"][:, start:end, :]]
      next_obs += [sample["next_obs"][:, start:end, :]]
    
    ## compute delta and normalize
    return self._process_sample({
      "obs": torch.cat(obs),
      "act": torch.cat(act),
      "next_obs": torch.cat(next_obs)
    })

  """
  """
  def _get_checkpoint(self, **kwargs) -> Dict[str, Any]:
    kwargs["model"] = self.model.state_dict()
    kwargs["optim"] = self.optim.state_dict()
    kwargs["buffer"] = self.buffer
    kwargs["normalizer"] = self.normalizer
    return kwargs

  """
  """
  @staticmethod
  def validate_params(params: Dict) -> None:
    QuickAdaptBase.validate_params(params)

    ## model IO shape
    obs_dim = params["env"].observation_space.shape[0]
    act_dim = params["env"].action_space.shape[0]
    inp_dim = obs_dim + act_dim

    assert params["model_params"]["input_dim"] == inp_dim
    assert params["model_params"]["output_dim"] == obs_dim

    ## GrBAL specific
    expected_n_episode_per_task = params["num_iter"] * params["num_episode_per_task"]
    assert params["num_future_obs"] >= params["mpc_horizon"]
    if expected_n_episode_per_task != params["buffer_params"]["buffer_size"]:
      warn_and_ask("""The expected total number of episodes per taskis %02d episodes.
      However, each buffer are expected to contain %02d episodes. Continue? (Y/N)""" %
      (expected_n_episode_per_task, params["buffer_params"]["buffer_size"]))