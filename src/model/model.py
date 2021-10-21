from typing import Dict, Type

import torch.nn as nn

REGISTERED_ACTIVATION_FUNC = {
  "tanh": nn.Tanh,
  "relu": nn.ReLU,
  "sigmoid": nn.Sigmoid,
  "softmax": nn.Softmax,
  "identity": nn.Identity
}

class Model(nn.Module):
  """
  """
  def __init__(self, input_dim: int, output_dim: int) -> None:
    super().__init__()
    self.input_dim : int = input_dim
    self.output_dim : int = output_dim
  
  """
  """
  @staticmethod
  def validate_params(params: Dict) -> None:
    assert params["input_dim"] > 0
    assert params["output_dim"] > 0

  """
  """
  @classmethod
  def instantiate_model(
    model_type: "Type[Model]",
    model_params: Dict) -> "Model":

    model_type.validate_params(model_params)
    return model_type(**model_params)