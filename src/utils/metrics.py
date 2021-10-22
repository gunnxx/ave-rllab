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