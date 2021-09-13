import pytest
import torch
import numpy as np

from src.buffer.consecutive_buffer import ConsecutiveBuffer
from src.utils.common import batch_data

def test_init_from_classmethod():
  # expected to raise an assertion
  with pytest.raises(AssertionError):
    _ = ConsecutiveBuffer.instantiate_buffer(
      {"buffer_size": 1, "consecutive_size": 2})
  
  # expected to success
  _ = ConsecutiveBuffer.instantiate_buffer(
    {"buffer_size": 1, "consecutive_size": 1})


def test_method_init_buffer():
  buffer_size = 3
  consecutive_size = 1
  
  buffer = ConsecutiveBuffer(buffer_size, consecutive_size)
  
  # _init_buffer() is called during the first call of store()
  buffer.store(
    key_1=0.356,
    key_2=[0.214],
    key_3=np.random.randn(2, 3),
    key_4=torch.randn(4, 5, 6))
  
  assert buffer.data_keys == ["key_1", "key_2", "key_3", "key_4"]
  assert buffer.data['key_1'].shape == (buffer_size, consecutive_size)
  assert buffer.data['key_2'].shape == (buffer_size, consecutive_size, 1)
  assert buffer.data['key_3'].shape == (buffer_size, consecutive_size, 2, 3)
  assert buffer.data['key_4'].shape == (buffer_size, consecutive_size, 4, 5, 6)


def test_method_store():
  buffer_size = 3
  consecutive_size = 2
  
  buffer = ConsecutiveBuffer(buffer_size, consecutive_size)

  # because consecutive_size = 2, the actual storing
  # operation starts at the 2-nd call of store()
  buffer.store(key_1=torch.randn(1))
  assert buffer.size == 0
  assert buffer.ptr  == 0
  assert torch.equal(
    buffer.data["key_1"],
    torch.zeros((buffer_size, consecutive_size, 1)))

  buffer.store(key_1=torch.randn(1))
  assert buffer.size == 1
  assert buffer.ptr  == 1
  assert not torch.equal(
    buffer.data["key_1"],
    torch.zeros((buffer_size, consecutive_size, 1)))


def test_method_sample():
  buffer_size = 5
  consecutive_size = 3
  
  buffer = ConsecutiveBuffer(buffer_size, consecutive_size)
  
  for _ in range(buffer_size + consecutive_size - 1):
    buffer.store(key_1=torch.randn(2, 3))

  batch_size = 1
  last_n = 3
  idx = [4, 0]

  assert buffer.sample_batch(batch_size)["key_1"].shape == \
    (batch_size, consecutive_size, 2, 3)
  assert buffer.sample_last_n(last_n)["key_1"].shape == \
    (last_n, consecutive_size, 2, 3)
  assert buffer.sample_idx(idx)["key_1"].shape == \
    (len(idx), consecutive_size, 2, 3)


def test_condition_circular_when_full_buffer():
  buffer_size = 5
  consecutive_size = 2
  
  buffer = ConsecutiveBuffer(buffer_size, consecutive_size)

  # full capacity
  data = []
  for _ in range(buffer_size + consecutive_size - 1):
    data.append(torch.randn(1, 2))
    buffer.store(data=data[-1])
  
  assert buffer.size == buffer_size
  assert buffer.ptr  == 0 # point to next slot in buffer
  assert torch.equal( # check oldest and newest content
    buffer.sample_idx([buffer.ptr, buffer.ptr - 1])["data"],
    batch_data( # batch-level
      batch_data( data[0],  data[1]), # consecutive-level
      batch_data(data[-2], data[-1])))
  assert torch.equal( # check n-newest content
    buffer.sample_last_n(2)["data"],
    batch_data( # batch-level
      batch_data(data[-3], data[-2]), # consecutive-level
      batch_data(data[-2], data[-1])))

  # add twice data overflows
  for _ in range(2):
    data.append(torch.randn(1, 2))
    buffer.store(data=data[-1])

  assert buffer.size == buffer_size
  assert buffer.ptr  == 2 # point to next slot in buffer
  assert torch.equal( # check oldest and newest content
    buffer.sample_idx([buffer.ptr, buffer.ptr - 1])["data"],
    batch_data( # batch-level
      batch_data( data[2],  data[3]), # consecutive-level
      batch_data(data[-2], data[-1])))
  assert torch.equal( # check n-newest content
    buffer.sample_last_n(2)["data"],
    batch_data( # batch-level
      batch_data(data[-3], data[-2]), # consecutive-level
      batch_data(data[-2], data[-1])))