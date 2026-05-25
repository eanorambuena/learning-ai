import numpy as np
from utils import init_and_train, identity, sigmoid, dIdentity, dSigmoid, forward
from visualize import plot_decision_boundary

def or_gate(train_func, activation, d_activation):
  print("\n=== OR ===")
  data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32)
  target = np.array([[0, 0], [1, 1], [1, 1], [1, 1]], dtype=np.float32)
  w, b = init_and_train(data, target, train_func, activation, d_activation)
  plot_decision_boundary(data, target, w, b, forward, activation, "OR_Gate")

def and_gate(train_func, activation, d_activation):
  print("\n=== AND ===")
  data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32)
  target = np.array([[0, 0], [0, 0], [0, 0], [1, 1]], dtype=np.float32)
  w, b = init_and_train(data, target, train_func, activation, d_activation)
  plot_decision_boundary(data, target, w, b, forward, activation, "AND_Gate")

def xor_gate(train_func, activation, d_activation):
  print("\n=== XOR ===")
  data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32)
  target = np.array([[0, 0], [1, 1], [1, 1], [0, 0]], dtype=np.float32)
  w, b = init_and_train(data, target, train_func, activation, d_activation)
  plot_decision_boundary(data, target, w, b, forward, activation, "XOR_Gate")

def test(train_func, activation=identity, d_activation=dIdentity):
  or_gate(train_func, activation, d_activation)
  and_gate(train_func, activation, d_activation)
  xor_gate(train_func, activation, d_activation)
