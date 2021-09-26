import torch.optim as optim

## For convenience only, so we can
## specify the optimizer from JSON
## through str datatype.
REGISTERED_OPTIM = {
  "adam": optim.Adam,
  "sgd": optim.SGD
}

## For convenience only, user only
## need to implement run().
class Algo:
  """
  """
  def run(self) -> None:
    raise NotImplementedError()