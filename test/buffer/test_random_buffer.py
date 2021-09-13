import pytest
import torch
import numpy as np

from src.buffer.random_buffer import RandomBuffer
from src.utils.common import batch_data

def test_init_from_classmethod():
  # expected to raise an assertion
  with pytest.raises(AssertionError):
    _ = RandomBuffer.instantiate_buffer({"buffer_size": 0})
  
  # expected to success
  _ = RandomBuffer.instantiate_buffer({"buffer_size": 1})


def test_method_init_buffer():
  buffer_size = 3
  buffer = RandomBuffer(buffer_size)

  # _init_buffer() is called during the first call of store()
  buffer.store(
    key_1=0.356,
    key_2=[0.123],
    key_3=np.random.randn(2, 3),
    key_4=torch.randn(4, 5, 6))
  
  assert buffer.data_keys == ["key_1", "key_2", "key_3", "key_4"]
  assert buffer.data['key_1'].shape == (buffer_size,)
  assert buffer.data['key_2'].shape == (buffer_size, 1)
  assert buffer.data['key_3'].shape == (buffer_size, 2, 3)
  assert buffer.data['key_4'].shape == (buffer_size, 4, 5, 6)


def test_method_store():
  buffer_size = 2
  buffer = RandomBuffer(buffer_size)

  assert buffer.size == 0
  assert buffer.ptr  == 0
  assert buffer.data == {}

  sample = torch.randn(4, 5)
  buffer.store(key_1=sample)

  assert buffer.size == 1
  assert buffer.ptr  == 1
  assert torch.equal(
    buffer.data["key_1"],
    batch_data(sample, torch.zeros(4, 5)))


def test_method_sample():
  buffer_size = 6
  buffer = RandomBuffer(buffer_size)

  for _ in range(buffer_size):
    buffer.store(data=torch.randn(2, 3))
  
  batch_sz = 1
  last_n = 2
  idx = [-1, 5, 0]

  assert buffer.sample_batch(batch_sz)["data"].shape == (batch_sz, 2, 3)
  assert buffer.sample_last_n(last_n)["data"].shape == (last_n, 2, 3)
  assert buffer.sample_idx(idx)["data"].shape == (len(idx), 2, 3)


def test_condition_circular_when_full_buffer():
  buffer_size = 6
  buffer = RandomBuffer(6)

  # full capacity
  data = []
  for _ in range(buffer_size):
    data.append(torch.randn(1, 2))
    buffer.store(key=data[-1])
  
  assert buffer.size == buffer_size
  assert buffer.ptr  == 0 # point to the next slot in buffer
  assert torch.equal( # check oldest and newest content
    buffer.sample_idx([buffer.ptr, buffer.ptr - 1])["key"],
    batch_data(data[0], data[-1]))
  assert torch.equal( # check n-newest content
    buffer.sample_last_n(2)["key"],
    batch_data(data[-2], data[-1]))
  
  # add twice data overflows
  for _ in range(2):
    data.append(torch.randn(1, 2))
    buffer.store(key=data[-1])
  
  assert buffer.size == buffer_size
  assert buffer.ptr  == 2 # point to the next slot in buffer
  assert torch.equal( # check oldest and newest content
    buffer.sample_idx([buffer.ptr, buffer.ptr - 1])["key"],
    batch_data(data[2], data[-1]))
  assert torch.equal( # check n-newest content
    buffer.sample_last_n(2)["key"],
    batch_data(data[-2], data[-1]))