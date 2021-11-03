from typing import Any, Dict, List, Tuple, Type

import numpy as np
import torch
import torch.nn as nn

from copy import deepcopy
from gym import Env
from torch import Tensor
from torch.optim import SGD
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

from src.algo.quick_adaptation.base import QuickAdaptBase
from src.buffer.buffer import Buffer
from src.model.deterministic_model import DeterministicModel
from src.utils.common import cast_to_torch, warn_and_ask
from src.utils.metrics import MovingAverage

class FAMLE(QuickAdaptBase):
  # List of constructor parameters along its default values.
  # This will be used to construct default config in run.py.
  REQUIRED_CONFIG_KEYS = {
    "env": None,
    "device": "cpu",
    "seed": 1501,
    "exp_dir": "exp/famle",
    "num_iter": 1,
    "num_random_iter": 1,
    "max_epoch_per_iter": 5000,
    "render_freq": 1,
    "batch_size": 512,
    "num_episode_per_task": 100,
    "max_episode_length": 100,
    "num_samples_to_adapt": 100,
    "buffer_type": "random_buffer",
    "buffer_params": {},
    "model_type": "deterministic_mlp_model",
    "model_params": {},
    "embedding_dim": 8,
    "mpc_horizon": 10,
    "mpc_num_trajectories": 4096,
    "task_learning_rate": 1e-4,
    "meta_model_learning_rate": 3e-1,
    "meta_embed_learning_rate": 3e-1,
    "num_inner_grad_steps": 10,
    "training_damages": [[1., 1.]],
    "testing_damages": [[1., 1.]]
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
    num_samples_to_adapt: int,
    buffer_type: Type[Buffer],
    buffer_params: Dict[str, Any],
    model_type: Type[DeterministicModel],
    model_params: Dict[str, Any],
    embedding_dim: int,
    mpc_horizon: int,
    mpc_num_trajectories: int,
    task_learning_rate: float,
    meta_model_learning_rate: float,
    meta_embed_learning_rate: float,
    num_inner_grad_steps: int,
    training_damages: List[List[float]],
    testing_damages: List[List[float]]) -> None:

    ## based on original FAMLE implementation
    ## https://github.com/resibots/kaushik_2020_famle
    is_store_per_episode = False
    validation_damages = []

    super().__init__(env, device, seed, exp_dir, num_iter,
    num_random_iter, max_epoch_per_iter, render_freq,
    batch_size, num_episode_per_task, max_episode_length,
    num_samples_to_adapt, mpc_horizon, is_store_per_episode,
    buffer_type, buffer_params, training_damages,
    validation_damages, testing_damages)

    ## training hyperparams
    self.task_learning_rate : float = task_learning_rate
    self.num_inner_grad_steps : int = num_inner_grad_steps
    self.loss_fn : nn.MSELoss = nn.MSELoss()

    ## MPC hyperparams
    self.mpc_num_trajectories : int = mpc_num_trajectories

    ## initialize model
    self.model : DeterministicModel = model_type.instantiate_model(model_params)
    self.model.to(self.device)
    
    self.embed : nn.Embedding = nn.Embedding(self.num_training_task, embedding_dim)
    self.embed.to(self.device)

    ## optimizer and scheduler
    self.model_optim : SGD = SGD(self.model.parameters(), meta_model_learning_rate)
    self.embed_optim : SGD = SGD(self.embed.parameters(), meta_embed_learning_rate)

    self.model_scheduler : ReduceLROnPlateau = ReduceLROnPlateau(self.model_optim, factor=.5)
    self.embed_scheduler : ReduceLROnPlateau = ReduceLROnPlateau(self.embed_optim, factor=.5)

    ## store (mean, std) of each dimension in the buffer
    self.normalizer : Dict[Tuple[Tensor]] = None
  
  """
  """
  def adapt_model(self, buffer: Dict[str, List[np.ndarray]]) -> Any:
    clone_embed = deepcopy(self.embed)
    clone_model = deepcopy(self.model)

    ## no adaptation, situation is randomly sampled
    if buffer["size"] == 0:
      task_idx = np.random.randint(0, self.num_training_task)
      task_idx = cast_to_torch([task_idx], torch.int32, self.device)
      return (clone_model, clone_embed(task_idx).squeeze())

    clone_embed.train()
    clone_model.train()

    ## prepare the clone optimizer
    clone_optim = SGD(
      list(clone_model.parameters()) + list(clone_embed.parameters()),
      self.task_learning_rate)
    
    ## prepare the input and pick embedding best represent current situation
    buffer = {k: cast_to_torch(v, torch.float32, self.device) for k, v in buffer.items()}
    buffer = self._process_sample(buffer)
    task_idx = self._estimate_task(buffer)
    batch_size = buffer["obs"].shape[0]
    embed_in = cast_to_torch([task_idx]*batch_size, torch.int32, self.device)

    ## inner SGD loop ~ adaptation
    for _ in range(self.num_inner_grad_steps):
      clone_model.zero_grad()
      clone_embed.zero_grad()

      ## forward and compute loss
      embed_ = clone_embed(embed_in)
      in_ = torch.cat([buffer["obs"], buffer["act"], embed_], dim=-1)
      diff_pred = clone_model(in_)
      diff_true = buffer["delta"]
      loss = self.loss_fn(diff_pred, diff_true)

      ## update
      loss.backward()
      clone_optim.step()
    
    task_idx = cast_to_torch([task_idx], torch.int32, self.device)
    return (clone_model, clone_embed(task_idx).squeeze())
  
  """
  """
  def mpc(self, model: Any, obs: np.ndarray) -> np.ndarray:
    model, embed_ = model
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

    ## (embed_dim) -> (mpc_num_trajectories, embed_dim)
    embed_ = embed_.unsqueeze(0).repeat(self.mpc_num_trajectories, 1)

    ## normalized all actions
    act = (act - self.normalizer["act"][0]) / (self.normalizer["act"][1] + 1e-10)

    ## rollout sampled actions
    with torch.no_grad():
      for i in range(self.mpc_horizon):
        ## normalized copy of current observation
        obs_ = deepcopy(obs) 
        obs_ = (obs_ - self.normalizer["obs"][0]) / (self.normalizer["obs"][1] + 1e-10)

        ## predict normalized-delta
        in_ = torch.cat([obs_, act[i], embed_], dim=-1)
        diff = model(in_)

        ## denormalize
        diff *= self.normalizer["delta"][1] + 1e-10
        diff += self.normalizer["delta"][0]

        ## unnormalized predicted next_obs
        next_obs = obs + diff

        ## compute reward: @staticmethod
        goal = torch.tensor(self.env.goal, device=self.device)
        reward = self.env.reward(obs, act[i], next_obs, goal)
        returns += reward

        ## update obs for next iteration
        obs = next_obs
    
    ## choose the first action of the best trajectory
    best_idx = torch.argmax(returns)
    return act[0, best_idx, :].cpu().numpy()
  
  """
  """
  def meta_learn_model(self, iter: int) -> None:
    moving_avg = MovingAverage()

    ## computed values are stored in `self.normalizer`
    self._compute_normalization_constant()

    ## iterate through every tasks one-by-one
    for epoch in tqdm(range(self.max_epoch_per_iter), "Training", position=0, leave=False):
      task_idx = epoch % self.num_training_task
      
      self.model.train()
      self.embed.train()

      ## prepare the clones for inner SGD loop
      clone_model = deepcopy(self.model)
      clone_embed = deepcopy(self.embed)
      clone_optim = SGD(
        list(clone_model.parameters()) + list(clone_embed.parameters()),
        self.task_learning_rate)

      ## prepare the input (requires_grad=False)
      sample = self.buffer[task_idx].sample_batch(self.batch_size)
      sample = self._process_sample(sample)
      embed_in = cast_to_torch([task_idx]*self.batch_size, torch.int32, self.device)
      
      ## inner SGD loop
      for _ in range(self.num_inner_grad_steps):
        clone_model.zero_grad()
        clone_embed.zero_grad()

        ## forward and compute loss
        embed_ = clone_embed(embed_in)
        in_ = torch.cat([sample["obs"], sample["act"], embed_], dim=-1)
        diff_pred = clone_model(in_)
        diff_true = sample["delta"]
        loss = self.loss_fn(diff_pred, diff_true)

        ## update
        loss.backward()
        clone_optim.step()
      
      ## REPTILE
      ## compute pseudo-meta-gradient
      model_meta_grad = []
      for cmp, mp in zip(clone_model.parameters(), self.model.parameters()):
        model_meta_grad += [cmp - mp]
      
      embed_meta_grad = []
      for cep, ep in zip(clone_embed.parameters(), self.embed.parameters()):
        embed_meta_grad += [cep - ep]
      
      ## fill the pseudo-meta-gradient to the parameters().grad
      for p, mg in zip(self.model.parameters(), model_meta_grad):
        p.grad = -mg
      
      for p, mg in zip(self.embed.parameters(), embed_meta_grad):
        p.grad = -mg
      
      ## meta-update
      self.model_optim.step()
      self.embed_optim.step()

      self.model.eval()
      self.embed.eval()

      ## evaluate on the training task itself
      with torch.no_grad():
        ## prepare the input
        sample = self.buffer[task_idx].sample_batch(self.batch_size)
        sample = self._process_sample(sample)
        embed_in = cast_to_torch([task_idx]*self.batch_size, torch.int32, self.device)
        
        ## forward and compute loss
        embed_ = self.embed(embed_in)
        in_ = torch.cat([sample["obs"], sample["act"], embed_], dim=-1)
        diff_pred = self.model(in_)
        diff_true = sample["delta"]
        loss = self.loss_fn(diff_pred, diff_true)

        ## compute moving average
        moving_avg(loss.item())
      
      ## scheduler update based on moving avg loss of all training tasks
      # if (epoch + 1) % self.num_training_task == 0:
      #   self.model_scheduler.step(moving_avg.data)
      #   self.embed_scheduler.step(moving_avg.data)
      
      log = {"iter": iter, "task": task_idx, "loss": loss.item(), "ema-loss": moving_avg.data}
      self.logger.store(training=log)

      ## log: stdout
      tqdm.write("Iter %02d Epoch %04d Task %02d || %.04f || %.04f" %
      (iter, epoch, task_idx, loss.item(), moving_avg.data))
  
  """
  """
  def handle_end_of_iteration(self, iter: int, time: float) -> None:
    self.logger.epoch_store(iter=iter, time=time)
    self.logger.dump()
    
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
    
    ## (total_datapoints, feature_sz)
    obs = torch.cat(obs)
    act = torch.cat(act)
    delta = torch.cat(delta)

    ## compute the normalizer
    self.normalizer = {
      'obs': (obs.mean(dim=0), obs.std(dim=0)),
      'act': (act.mean(dim=0), act.std(dim=0)),
      'delta': (delta.mean(dim=0), delta.std(dim=0))
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
  def _estimate_task(self, buffer: Dict[str, Tensor]) -> float:
    buffer_sz = buffer['obs'].shape[0]
    embed_in_ = list(range(self.num_training_task))

    self.model.eval()
    self.embed.eval()

    ## task == maximum likelihood ~ minimum MSE
    with torch.no_grad():
      _repeat = (self.num_training_task, 1, 1)
      _reshape1 = (self.num_training_task * buffer_sz, -1)
      _reshape2 = (self.num_training_task, buffer_sz, -1)

      ## (buffer_sz, feature_sz) -> (training_task, buffer_sz, feature_sz)
      buffer = {k: v.unsqueeze(0).repeat(*_repeat) for k, v in buffer.items()}

      ## (training_task, embed_dim) -> (training_task, buffer_sz, embed_dim)
      embed_ = self.embed(cast_to_torch(embed_in_, torch.int32, self.device))
      embed_ = embed_.unsqueeze(1).repeat(1, buffer_sz, 1)

      ## (training_task, buffer_sz, ..) -> (training_task * buffer_sz, ..)
      in_ = torch.cat([buffer["obs"], buffer["act"], embed_], dim=-1)
      in_ = in_.reshape(*_reshape1)

      diff_pred = self.model(in_)
      diff_true = buffer["delta"].reshape(*_reshape1)

      ## compute loss
      loss = torch.pow(diff_pred - diff_true, 2).sum(-1)
      loss = loss.reshape(*_reshape2).sum([-2, -1])

      assert loss.shape == (self.num_training_task,)
      task_idx = torch.argmin(loss)
    return task_idx
  
  """
  """
  def _get_checkpoint(self, **kwargs) -> Dict[str, Any]:
    kwargs["model"] = self.model.state_dict()
    kwargs["embed"] = self.embed.state_dict()
    kwargs["model_optim"] = self.model_optim.state_dict()
    kwargs["embed_optim"] = self.embed_optim.state_dict()
    kwargs["model_scheduler"] = self.model_scheduler.state_dict()
    kwargs["embed_scheduler"] = self.embed_scheduler.state_dict()
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
    inp_dim = obs_dim + act_dim + params["embedding_dim"]
    
    assert params["model_params"]["input_dim"] == inp_dim
    assert params["model_params"]["output_dim"] == obs_dim

    ## based on original FAMLE hyperparams
    assert params["max_epoch_per_iter"] % len(params["training_damages"]) == 0
    if params["num_iter"] != 1:
      warn_and_ask("""Original FAMLE implementation uses only `num_iter = 1`.
      Currently, `num_iter = %d`. Continue? (Y/N)""" % params["num_iter"])