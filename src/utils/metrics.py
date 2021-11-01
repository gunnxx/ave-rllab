from typing import List

"""
"""
class MovingAverage:
  def __init__(self, decay: float = 0.99) -> None:
    self.decay : float = decay
    self.data_ : float = None
  
  @property
  def data(self) -> float:
    return self.data_
  
  def __call__(self, new_data : float) -> None:
    if self.data_ is None: self.data_ = new_data
    self.data_ = self.data_ * self.decay + new_data * (1 - self.decay)

"""
"""
class EarlyStopMetric:
  def __init__(self, patience: int = 2, warmup: int = 10, decay: float = 0.99) -> None:
    self.patience : int = patience
    self.warmup : int = warmup
    self.iteration : int = 0

    ## container for last `self.patience + 1` (moving) average
    self.metrics : List[float] = []
    self.ema : MovingAverage = MovingAverage(decay)
  
  def __call__(self, new_metric: float):
    ## compute the moving average of incoming data
    self.ema(new_metric)

    ## save the last `self.patience + 1` (moving) average
    self.metrics.append(self.ema.data)
    if len(self.metrics) > (self.patience + 1):
      self.metrics.pop(0)
    
    self.iteration += 1
  
  def is_stop(self) -> bool:
    ## return False in warmup phase
    if self.iteration < self.warmup:
      return False
    
    ## if there is progress among the last `self.patience` data excluding last element
    for metric in self.metrics[:-1]:
      if self.metrics[-1] < metric:
        return False
    
    ## no progress
    return True