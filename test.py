import numpy as np
from utils import init_and_train, identity, sigmoid, dIdentity, dSigmoid

def or_gate(train_func, activation, d_activation):
  print("\n=== OR ===")
  data: np.ndarray = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32)
  target: np.ndarray = np.array([[0, 0], [1, 1], [1, 1], [1, 1]], dtype=np.float32)
  init_and_train(data, target, train_func, activation, d_activation)

def and_gate(train_func, activation, d_activation):
  print("\n=== AND ===")
  data: np.ndarray = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32)
  target: np.ndarray = np.array([[0, 0], [0, 0], [0, 0], [1, 1]], dtype=np.float32)
  init_and_train(data, target, train_func, activation, d_activation)

def xor_gate(train_func, activation, d_activation):
  print("\n=== XOR ===")
  data: np.ndarray = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32)
  target: np.ndarray = np.array([[0, 0], [1, 1], [1, 1], [0, 0]], dtype=np.float32)
  init_and_train(data, target, train_func, activation, d_activation)

def test(train_func, activation=identity, d_activation=dIdentity):
  or_gate(train_func, activation, d_activation)
  and_gate(train_func, activation, d_activation)
  xor_gate(train_func, activation, d_activation)
