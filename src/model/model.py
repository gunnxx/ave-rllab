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
  @classmethod
  def instantiate_model(
    model_type: "Type[Model]",
    model_params: Dict) -> "Model":

    model_type.validate_params(model_params)
    return model_type(**model_params)