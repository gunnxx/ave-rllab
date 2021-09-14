from typing import Dict, Type

import torch
from learn2learn import clone_module, update_module

from adaptive_learner import AdaptiveLearner

class OrigGrBAL(AdaptiveLearner):
  """
  """
  @staticmethod
  def validate_config(config: Dict) -> None:
    pass

  """
  """
  def _model_meta_learn(self):
    # batch_size is the number of tasks
    # samples = (batch_sz, num_past_obs + num_future_obs, feature_sz)
    samples = self.buffer.sample_batch(self.batch_size)
    
    obs = self.sample["obs"]
    obs_support = obs[:, :self.num_past_obs, :]
    obs_query   = obs[:, self.num_past_obs:, :]

    act = self.sample["act"]
    act_support = act[:, :self.num_past_obs, :]
    act_query   = act[:, self.num_past_obs:, :]

    next_obs = self.sample["next_obs"]
    next_obs_support = next_obs[:, :self.num_past_obs, :]
    next_obs_query   = next_obs[:, self.num_past_obs:, :]

    # zero-ing out gradients and set to training mode
    self.model_dynamics.zero_grad()
    self.model_dynamics.train()

    # training
    meta_grads = tuple(torch.zeros_like(p)
      for p in self.model_dynamics.parameters())

    for i in range(self.batch_size):
      clone_model_dynamics = clone_module(self.model_dynamics)

      # inner grad-descent loop (task training)
      for _ in range(self.num_inner_grad_steps):
        log_prob_model = clone_model_dynamics.log_prob_from_data(
          torch.cat([obs_support[i], act_support[i]]),
          next_obs_support[i])
        
        grad = torch.autograd.grad(
          -log_prob_model.sum(),
          clone_model_dynamics.parameters(),
          create_graph=True)
        
        clone_model_dynamics = update_module(
          clone_model_dynamics,
          updates=tuple(self.task_learning_rate*g for g in grad))
      
      # meta-gradient
      log_prob_model = clone_model_dynamics.log_prob_from_data(
        torch.cat([obs_query[i], act_query[i]]),
        next_obs_query[i])
      
      grad = torch.autograd.grad(
        -log_prob_model.sum(),
        self.model_dynamics.parameters())
      
      meta_grads = tuple(mg + g for mg, g in zip(meta_grads, grad))
    
    # meta-update
    for p, mg in zip(self.model_dynamics.parameters(), meta_grads):
      p.grad = mg/self.batch_size
    self.meta_optimizer.step()