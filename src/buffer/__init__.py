from src.buffer.random_buffer import RandomBuffer
from src.buffer.consecutive_buffer import ConsecutiveBuffer

REGISTERED_BUFFER = {
  "random_buffer": RandomBuffer,
  "consecutive_buffer": ConsecutiveBuffer
}