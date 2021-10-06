import time

class Timer:
  """
  Taken from https://github.com/kevinzakka/torchkit.
  """
  def __init__(self) -> None:
    self.reset()
  
  """
  """
  def elapsed(self) -> float:
    return time.time() - self.time
  
  """
  """
  def reset(self) -> None:
    self.time = time.time()