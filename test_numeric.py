import numpy as np
from utils import init_and_train, identity, dIdentity

def divmod_gate(train_func, forward_func, activation=identity, d_activation=dIdentity):
  print("\n=== DIVMOD (quotient, remainder) ===")
  data: np.ndarray = np.array([
    [2, 2], [3, 2], [4, 2], [5, 2],
    [6, 3], [7, 3], [8, 3], [9, 3],
    [10, 4], [11, 4], [12, 4], [13, 4],
  ], dtype=np.float32)
  target: np.ndarray = np.array([
    [1, 0], [1, 1], [2, 0], [2, 1],
    [2, 0], [2, 1], [2, 2], [3, 0],
    [2, 2], [2, 3], [3, 0], [3, 1],
  ], dtype=np.float32)
  init_and_train(data, target, train_func, forward_func, activation, d_activation)

def add_gate(train_func, forward_func, activation=identity, d_activation=dIdentity):
  print("\n=== ADD (a + b, a - b) ===")
  data: np.ndarray = np.array([
    [1, 2], [2, 3], [3, 1], [4, 2],
    [5, 3], [6, 1], [7, 4], [2, 5],
    [8, 2], [3, 6], [4, 4], [1, 7],
  ], dtype=np.float32)
  target: np.ndarray = np.array([
    [3, -1], [5, -1], [4, 2], [6, 2],
    [8, 2], [7, 5], [11, 3], [7, -3],
    [10, 6], [9, -3], [8, 0], [8, -6],
  ], dtype=np.float32)
  init_and_train(data, target, train_func, forward_func, activation, d_activation)

def maxmin_gate(train_func, forward_func, activation=identity, d_activation=dIdentity):
  print("\n=== MAXMIN (max, min) ===")
  data: np.ndarray = np.array([
    [3, 1], [2, 5], [4, 2], [1, 6],
    [7, 3], [5, 8], [2, 2], [9, 1],
    [4, 6], [3, 7], [8, 4], [6, 6],
  ], dtype=np.float32)
  target: np.ndarray = np.array([
    [3, 1], [5, 2], [4, 2], [6, 1],
    [7, 3], [8, 5], [2, 2], [9, 1],
    [6, 4], [7, 3], [8, 4], [6, 6],
  ], dtype=np.float32)
  init_and_train(data, target, train_func, forward_func, activation, d_activation)

def prod_gate(train_func, forward_func, activation=identity, d_activation=dIdentity):
  print("\n=== PROD (a * b, a // b) ===")
  data: np.ndarray = np.array([
    [2, 3], [4, 2], [3, 2], [5, 2],
    [6, 3], [4, 4], [5, 3], [7, 2],
    [3, 5], [6, 4], [8, 2], [4, 5],
  ], dtype=np.float32)
  target: np.ndarray = np.array([
    [6, 0], [8, 2], [6, 1], [10, 3],
    [18, 2], [16, 1], [15, 2], [14, 5],
    [15, 0], [24, 2], [16, 4], [20, 0],
  ], dtype=np.float32)
  init_and_train(data, target, train_func, forward_func, activation, d_activation)

def test_numeric(train_func, forward_func, activation=identity, d_activation=dIdentity):
  divmod_gate(train_func, forward_func, activation, d_activation)
  add_gate(train_func, forward_func, activation, d_activation)
  maxmin_gate(train_func, forward_func, activation, d_activation)
  prod_gate(train_func, forward_func, activation, d_activation)
