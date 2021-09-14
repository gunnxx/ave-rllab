from typing import Dict, Type

import torch.nn as nn

class Model(nn.Module):
  """
  """
  @classmethod
  def instantiate_model(
    model_type: "Type[Model]",
    model_params: Dict) -> "Model":

    model_type.validate_params(model_params)
    return model_type(**model_params)