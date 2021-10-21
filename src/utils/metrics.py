"""
"""
class MovingAverage:
  def __init__(self, decay: float = 0.99) -> None:
    self.decay : float = decay
    self.data : float = 0.
  
  @property
  def data(self) -> float:
    return self.data
  
  def __call__(self, new_data : float) -> None:
    if self.data is None: self.data = new_data
    self.data = self.data * self.decay + new_data * (1 - self.decay)