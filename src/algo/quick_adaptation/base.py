from typing import Any, Dict, List, Type

import gym
import numpy as np
import torch
from tqdm import tqdm

from src.algo.algo import Algo
from src.buffer.buffer import Buffer
from src.utils.logger import Logger
from src.utils.timer import Timer

class QuickAdaptBase(Algo):
  """
  """
  def __init__(self,
    env: gym.Env,
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
    mpc_horizon: int,
    is_store_per_episode: bool,
    buffer_type: Type[Buffer],
    buffer_params: Dict[str, Any],
    training_damages: List[List[float]],
    validation_damages: List[List[float]],
    testing_damages: List[List[float]] = []) -> None:
    
    ## general variables
    self.env : gym.Env = env
    self.device : torch.device = torch.device(device)
    self.logger : Logger = Logger(exp_dir)

    ## set random seed
    torch.manual_seed(seed)
    np.random.seed(seed)

    self.env.seed(seed)
    self.env.action_space.seed(seed)
    self.env.observation_space.seed(seed)
    self.env.reset()

    ## training and data collection hyperparams
    self.num_iter : int = num_iter
    self.num_random_iter : int = num_random_iter
    self.max_epoch_per_iter : int = max_epoch_per_iter
    self.render_freq: int = render_freq
    self.batch_size : int = batch_size
    self.num_samples_to_adapt : int = num_samples_to_adapt
    self.mpc_horizon : int = mpc_horizon
    self.is_store_per_episode : bool = is_store_per_episode
    
    self.num_training_task : int = len(training_damages)
    self.training_damages : List[List[float]] = training_damages

    self.num_validation_task : int = len(validation_damages)
    self.validation_damages : List[List[float]] = validation_damages

    self.num_testing_task : int = len(testing_damages)
    self.testing_damages : List[List[float]] = testing_damages

    self.num_episode_per_task : int = num_episode_per_task
    self.max_episode_length : int = max_episode_length

    ## instantiate buffer for each task
    self.buffer : List[Buffer] = [buffer_type.instantiate_buffer(buffer_params)
      for _ in range(self.num_training_task + self.num_validation_task)]
  

  """
  """
  def run(self) -> None:
    timer = Timer()

    for iter in tqdm(range(self.num_iter), "Iteration", position=1):
      tqdm.write("Iteration %03d" % iter)

      # ----------------- DATA COLLECTION ----------------- #

      timer.reset()
      
      log = dict(iter=iter)
      all_damages = self.training_damages + self.validation_damages
      for task_idx in tqdm(range(len(all_damages)), "Collect data", position=0, leave=False):
        for episode_idx in range(self.num_episode_per_task):
          obs = self.env.reset(actuator_damage=all_damages[task_idx])
          episode_ret = 0
          episode_len = 0

          ## true_buffer -> self.buffer if self.is_store_per_episode
          ## temp_buffer -> self.model_adaptation()
          true_buffer = self._get_dict_as_buffer(self.max_episode_length)
          temp_buffer = self._get_dict_as_buffer(self.num_samples_to_adapt)

          ## rollout: terminal state depends on self.is_store_per_episode
          for _ in range(self.max_episode_length):
            ## start with random policy, MPC afterwards
            if iter < self.num_random_iter:
              act = self.env.action_space.sample()
            else:
              adapted_model = self.adapt_model(temp_buffer)
              act = self.mpc(adapted_model, obs)
            
            ## step in the environment
            next_obs, rew, done, info = self.env.step(act)
            success = info.get('success', False)

            ## store to temp and true buffer
            data = dict(obs=obs, act=act, next_obs=next_obs, rew=rew)
            self._fill_dict_buffer(true_buffer, data)
            self._fill_dict_buffer(temp_buffer, data)

            ## store to buffer each transition
            if not self.is_store_per_episode:
              self.buffer[task_idx].store(obs=obs,
              act=act, next_obs=next_obs, rew=rew)

            ## update values
            obs = next_obs
            episode_ret += rew
            episode_len += 1

            ## terminal state
            if (done or success) and (not self.is_store_per_episode):
              break
          
          ## store to buffer full one episode
          if self.is_store_per_episode:
            self.buffer[task_idx].store(
              obs = np.array(true_buffer['obs']),
              act = np.array(true_buffer['act']),
              next_obs = np.array(true_buffer['next_obs']),
              rew = np.array(true_buffer['rew']),
            )
          
          ## log return and episode length
          if task_idx < self.num_training_task:
            epoch_log = {"train_ep_ret": episode_ret, "train_ep_len": episode_len}
          else:
            epoch_log = {"valid_ep_ret": episode_ret, "valid_ep_len": episode_len}
          self.logger.epoch_store(**epoch_log)
          
          ## detailed log of return
          log_label = "task.%02d.%02d" % (task_idx, episode_idx)
          log[log_label] = episode_ret

      self.logger.store(data_collection=log)
        
      ## log: stdout
      n = len(all_damages) * self.num_episode_per_task * self.max_episode_length
      collection_time = timer.elapsed()
      tqdm.write("Collecting %d datapoints take %.02f s." % (n, collection_time))
    
      # -------------- MODEL TRAINING AND EVALUATION -------------- #

      self.meta_learn_model(iter=iter)

      ## log: stdout
      total = self.buffer[0].size * len(self.buffer)
      train_time = timer.elapsed() - collection_time
      tqdm.write("Training on %d datapoints for a maximum of %d iteration \
        (epoch) with a batch size of %d/iteration takes %.02f s." % 
      (total, self.max_epoch_per_iter, self.batch_size, train_time))

      ## evaluate
      log = dict(iter=iter)
      test_damages = self.validation_damages + self.testing_damages
      # for task_idx, task in enumerate(test_damages):
      for task_idx in tqdm(range(len(test_damages)), "Testing", position=0, leave=False):
        obs = self.env.reset(actuator_damage=test_damages[task_idx])
        episode_ret = 0
        episode_len = 0
        temp_buffer = self._get_dict_as_buffer(self.num_samples_to_adapt)

        ## setup render
        do_render = (iter + 1) % self.render_freq == 0
        if do_render:
          frames = [self.env.render("rgb_array",
          labels={'task id': '%02d' % task_idx, 'ret': 0, 'len': 0})]
        
        ## rollout: considering terminal state
        while(True):
          ## MPC
          adapted_model = self.adapt_model(temp_buffer)
          act = self.mpc(adapted_model, obs)

          ## step in the environment
          next_obs, rew, done, info = self.env.step(act)
          success = info.get('success', False)

          ## store to temp_buffer
          data = dict(obs=obs, act=act, next_obs=next_obs, rew=rew)
          self._fill_dict_buffer(temp_buffer, data)

          ## update values
          obs = next_obs
          episode_ret += rew
          episode_len += 1

          ## render
          if do_render:
            frames += [self.env.render("rgb_array", labels={'task id':
            '%02d' % task_idx, 'ret': episode_ret, 'len': episode_len})]

          ## terminal state
          if done or success or episode_len == self.max_episode_length:
            self.logger.epoch_store(test_ep_ret=episode_ret, test_ep_len=episode_len)
            break
        
        ## detailed log of return
        log_label = "task.%02d." % task_idx
        log[log_label + "ret"] = episode_ret
        log[log_label + "len"] = episode_len

        ## render to gif
        if do_render:
          self.logger.save_as_gif(frames,
          "iter%03d.task%02d.gif" % (iter, task_idx), fps=30)
      
      self.logger.store(evaluation=log)
      
      # -------------- END OF ITERATION: LOGGING -------------- #
        
      self.handle_end_of_iteration(iter, timer.elapsed())
    
    self.logger.close()

  """
  """
  def adapt_model(self, buffer: Dict[str, List[np.ndarray]]) -> Any:
    raise NotImplementedError()
  
  """
  """
  def mpc(self, model: Any, obs: np.ndarray)-> np.ndarray:
    raise NotImplementedError()
  
  """
  """
  def meta_learn_model(self, iter: int) -> None:
    raise NotImplementedError()
  
  """
  """
  def handle_end_of_iteration(self, iter: int, time: float) -> None:
    raise NotImplementedError()
  
  """
  """
  def _get_dict_as_buffer(self, max_size: int) -> Dict:
    return dict(obs=[], act=[], next_obs=[], rew=[], size=0, max_size=max_size)
  
  """
  """
  def _fill_dict_buffer(self, buffer: Dict, data: Dict) -> None:
    buffer['obs'].append(data['obs'])
    buffer['act'].append(data['act'])
    buffer['next_obs'].append(data['next_obs'])
    buffer['rew'].append(data['rew'])
    buffer['size'] += 1

    ## by design, true_buffer does not go here unlike temp_buffer
    if buffer['size'] > buffer['max_size']:
      buffer['obs'].pop(0)
      buffer['act'].pop(0)
      buffer['next_obs'].pop(0)
      buffer['rew'].pop(0)
      buffer['size'] -= 1

  """
  """
  @staticmethod
  def validate_params(params: Dict) -> None:
    assert params["num_random_iter"] <= params["num_iter"]
    assert params["num_iter"] % params["render_freq"] == 0
    assert params["buffer_params"]["device"] == params["device"]