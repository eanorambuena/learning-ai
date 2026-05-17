import numpy as np
from utils import init_and_train, identity, dIdentity

def divmod_gate(train_func, forward_func, activation=identity, d_activation=dIdentity):
  print("\n=== DIVMOD (quotient, remainder) ===")
  data: np.ndarray = np.array([[2, 2], [3, 2], [4, 2], [5, 2]], dtype=np.float32)
  target: np.ndarray = np.array([[1, 0], [1, 1], [2, 0], [2, 1]], dtype=np.float32)
  init_and_train(data, target, train_func, forward_func, activation, d_activation)

def test_numeric(train_func, forward_func, activation=identity, d_activation=dIdentity):
  divmod_gate(train_func, forward_func, activation, d_activation)