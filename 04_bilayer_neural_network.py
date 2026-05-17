#!/home/eanorambuena/miniconda/envs/learning-ai/bin/python
"""
Bilayer neural network implementation.
This file was generated with AI assistance.
"""
from numba import njit, prange
import numpy as np
from utils import sigmoid, dSigmoid, identity, dIdentity
from visualize import plot_bilayer_boundary

INPUT_N: np.int32 = 2
HIDDEN_N: np.int32 = 4
OUTPUT_N: np.int32 = 2
LEARNING_RATE: np.float32 = 0.5
EPOCHS: np.int32 = 5000

@njit
def init_params_bilayer():
  w1 = np.random.rand(HIDDEN_N, INPUT_N).astype(np.float32)
  b1 = np.random.rand(HIDDEN_N).astype(np.float32)
  w2 = np.random.rand(OUTPUT_N, HIDDEN_N).astype(np.float32)
  b2 = np.random.rand(OUTPUT_N).astype(np.float32)
  return w1, b1, w2, b2

@njit(parallel=True)
def forward_bilayer(data: np.ndarray, w1: np.ndarray, b1: np.ndarray, w2: np.ndarray, b2: np.ndarray, activation) -> np.ndarray:
  SAMPLES, INPUT_N = data.shape
  hidden = np.zeros((SAMPLES, HIDDEN_N), dtype=np.float32)
  output = np.zeros((SAMPLES, OUTPUT_N), dtype=np.float32)
  for i in prange(SAMPLES):
    for j in range(HIDDEN_N):
      hidden[i, j] = b1[j]
      for k in range(INPUT_N):
        hidden[i, j] += w1[j, k] * data[i, k]
    hidden[i] = activation(hidden[i])
    for j in range(OUTPUT_N):
      output[i, j] = b2[j]
      for k in range(HIDDEN_N):
        output[i, j] += w2[j, k] * hidden[i, k]
    output[i] = activation(output[i])
  return output

@njit(parallel=True)
def compute_gradients_bilayer(data: np.ndarray, target: np.ndarray, w1: np.ndarray, b1: np.ndarray, w2: np.ndarray, b2: np.ndarray, dw1: np.ndarray, db1: np.ndarray, dw2: np.ndarray, db2: np.ndarray, activation, d_activation) -> None:
  SAMPLES, INPUT_N = data.shape
  hidden = np.zeros((SAMPLES, HIDDEN_N), dtype=np.float32)
  hidden_z = np.zeros((SAMPLES, HIDDEN_N), dtype=np.float32)
  output_z = np.zeros((SAMPLES, OUTPUT_N), dtype=np.float32)
  for i in prange(SAMPLES):
    for j in range(HIDDEN_N):
      hidden_z[i, j] = b1[j]
      for k in range(INPUT_N):
        hidden_z[i, j] += w1[j, k] * data[i, k]
    hidden[i] = activation(hidden_z[i])
    for j in range(OUTPUT_N):
      output_z[i, j] = b2[j]
      for k in range(HIDDEN_N):
        output_z[i, j] += w2[j, k] * hidden[i, k]
    output = activation(output_z[i])
    d_output = d_activation(output_z[i])
    error_output = (output - target[i]) * d_output
    for j in range(OUTPUT_N):
      for k in range(HIDDEN_N):
        dw2[i, j, k] = error_output[j] * hidden[i, k]
      db2[i, j] = error_output[j]
    d_hidden = d_activation(hidden_z[i])
    error_hidden = np.zeros(HIDDEN_N, dtype=np.float32)
    for j in range(HIDDEN_N):
      for k in range(OUTPUT_N):
        error_hidden[j] += w2[k, j] * error_output[k]
      error_hidden[j] *= d_hidden[j]
    for j in range(HIDDEN_N):
      for k in range(INPUT_N):
        dw1[i, j, k] = error_hidden[j] * data[i, k]
      db1[i, j] = error_hidden[j]

@njit
def mean_reduce_2d(arr: np.ndarray) -> np.ndarray:
  SAMPLES, N, M = arr.shape
  mean = np.zeros((N, M), dtype=np.float32)
  for j in prange(N):
    for k in range(M):
      total = 0.0
      for i in range(SAMPLES):
        total += arr[i, j, k]
      mean[j, k] = total / SAMPLES
  return mean

@njit
def mean_reduce_1d_bilayer(arr: np.ndarray) -> np.ndarray:
  SAMPLES, N = arr.shape
  mean = np.zeros(N, dtype=np.float32)
  for j in prange(N):
    total = 0.0
    for i in range(SAMPLES):
      total += arr[i, j]
    mean[j] = total / SAMPLES
  return mean

@njit
def gradient_descent_bilayer(data: np.ndarray, target: np.ndarray, w1: np.ndarray, b1: np.ndarray, w2: np.ndarray, b2: np.ndarray, activation, d_activation) -> None:
  SAMPLES, INPUT_N = data.shape
  dw1 = np.zeros((SAMPLES, HIDDEN_N, INPUT_N), dtype=np.float32)
  db1 = np.zeros((SAMPLES, HIDDEN_N), dtype=np.float32)
  dw2 = np.zeros((SAMPLES, OUTPUT_N, HIDDEN_N), dtype=np.float32)
  db2 = np.zeros((SAMPLES, OUTPUT_N), dtype=np.float32)
  for epoch in range(EPOCHS):
    compute_gradients_bilayer(data, target, w1, b1, w2, b2, dw1, db1, dw2, db2, activation, d_activation)
    dw1_mean = mean_reduce_2d(dw1)
    db1_mean = mean_reduce_1d_bilayer(db1)
    dw2_mean = mean_reduce_2d(dw2)
    db2_mean = mean_reduce_1d_bilayer(db2)
    w1[:] = w1 - LEARNING_RATE * dw1_mean
    b1[:] = b1 - LEARNING_RATE * db1_mean
    w2[:] = w2 - LEARNING_RATE * dw2_mean
    b2[:] = b2 - LEARNING_RATE * db2_mean

def train_bilayer_wrapper(data, target, w, b, activation, d_activation):
  w1, b1, w2, b2 = init_params_bilayer()
  gradient_descent_bilayer(data, target, w1, b1, w2, b2, activation, d_activation)
  w[0] = w1
  w[1] = w2
  b[0] = b1
  b[1] = b2

def forward_bilayer_wrapper(data, w, b, activation):
  w1, w2 = w
  b1, b2 = b
  return forward_bilayer(data, w1, b1, w2, b2, activation)

def or_gate_bilayer():
  print("\n=== OR (Bilayer) ===")
  data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32)
  target = np.array([[0, 0], [1, 1], [1, 1], [1, 1]], dtype=np.float32)
  w1, b1, w2, b2 = init_params_bilayer()
  gradient_descent_bilayer(data, target, w1, b1, w2, b2, sigmoid, dSigmoid)
  prediction = forward_bilayer(data, w1, b1, w2, b2, sigmoid)
  print(f"prediction: {prediction}")
  print(f"target: {target}")
  plot_bilayer_boundary(data, target, w1, b1, w2, b2, sigmoid, "OR_Bilayer")

def and_gate_bilayer():
  print("\n=== AND (Bilayer) ===")
  data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32)
  target = np.array([[0, 0], [0, 0], [0, 0], [1, 1]], dtype=np.float32)
  w1, b1, w2, b2 = init_params_bilayer()
  gradient_descent_bilayer(data, target, w1, b1, w2, b2, sigmoid, dSigmoid)
  prediction = forward_bilayer(data, w1, b1, w2, b2, sigmoid)
  print(f"prediction: {prediction}")
  print(f"target: {target}")
  plot_bilayer_boundary(data, target, w1, b1, w2, b2, sigmoid, "AND_Bilayer")

def xor_gate_bilayer():
  print("\n=== XOR (Bilayer) ===")
  data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32)
  target = np.array([[0, 0], [1, 1], [1, 1], [0, 0]], dtype=np.float32)
  w1, b1, w2, b2 = init_params_bilayer()
  gradient_descent_bilayer(data, target, w1, b1, w2, b2, sigmoid, dSigmoid)
  prediction = forward_bilayer(data, w1, b1, w2, b2, sigmoid)
  print(f"prediction: {prediction}")
  print(f"target: {target}")
  plot_bilayer_boundary(data, target, w1, b1, w2, b2, sigmoid, "XOR_Bilayer")

def test_bilayer():
  or_gate_bilayer()
  and_gate_bilayer()
  xor_gate_bilayer()

def add_gate_bilayer():
  print("\n=== ADD (Bilayer) ===")
  data = np.array([
    [1, 2], [2, 3], [3, 1], [4, 2],
    [5, 3], [6, 1], [7, 4], [2, 5],
    [8, 2], [3, 6], [4, 4], [1, 7],
  ], dtype=np.float32)
  target = np.array([
    [3, -1], [5, -1], [4, 2], [6, 2],
    [8, 2], [7, 5], [11, 3], [7, -3],
    [10, 6], [9, -3], [8, 0], [8, -6],
  ], dtype=np.float32)
  w1, b1, w2, b2 = init_params_bilayer()
  gradient_descent_bilayer(data, target, w1, b1, w2, b2, identity, dIdentity)
  prediction = forward_bilayer(data, w1, b1, w2, b2, identity)
  print(f"prediction: {prediction}")
  print(f"target: {target}")
  plot_bilayer_boundary(data, target, w1, b1, w2, b2, identity, "ADD_Bilayer")

def maxmin_gate_bilayer():
  print("\n=== MAXMIN (Bilayer) ===")
  data = np.array([
    [3, 1], [2, 5], [4, 2], [1, 6],
    [7, 3], [5, 8], [2, 2], [9, 1],
    [4, 6], [3, 7], [8, 4], [6, 6],
  ], dtype=np.float32)
  target = np.array([
    [3, 1], [5, 2], [4, 2], [6, 1],
    [7, 3], [8, 5], [2, 2], [9, 1],
    [6, 4], [7, 3], [8, 4], [6, 6],
  ], dtype=np.float32)
  w1, b1, w2, b2 = init_params_bilayer()
  gradient_descent_bilayer(data, target, w1, b1, w2, b2, identity, dIdentity)
  prediction = forward_bilayer(data, w1, b1, w2, b2, identity)
  print(f"prediction: {prediction}")
  print(f"target: {target}")
  plot_bilayer_boundary(data, target, w1, b1, w2, b2, identity, "MAXMIN_Bilayer")

def prod_gate_bilayer():
  print("\n=== PROD (Bilayer) ===")
  data = np.array([
    [2, 3], [4, 2], [3, 2], [5, 2],
    [6, 3], [4, 4], [5, 3], [7, 2],
    [3, 5], [6, 4], [8, 2], [4, 5],
  ], dtype=np.float32)
  target = np.array([
    [6, 0], [8, 2], [6, 1], [10, 3],
    [18, 2], [16, 1], [15, 2], [14, 5],
    [15, 0], [24, 2], [16, 4], [20, 0],
  ], dtype=np.float32)
  w1, b1, w2, b2 = init_params_bilayer()
  gradient_descent_bilayer(data, target, w1, b1, w2, b2, identity, dIdentity)
  prediction = forward_bilayer(data, w1, b1, w2, b2, identity)
  print(f"prediction: {prediction}")
  print(f"target: {target}")
  plot_bilayer_boundary(data, target, w1, b1, w2, b2, identity, "PROD_Bilayer")

def test_numeric_bilayer():
  add_gate_bilayer()
  maxmin_gate_bilayer()
  prod_gate_bilayer()

def circle_gate_bilayer():
  print("\n=== CIRCLE (inner vs outer) ===")
  np.random.seed(42)
  inner_radius = 0.4
  outer_radius = 0.8
  num_inner = 30
  num_outer = 60
  data_list = []
  target_list = []
  for _ in range(num_inner):
    angle = np.random.uniform(0, 2 * np.pi)
    r = np.random.uniform(0, inner_radius)
    x = r * np.cos(angle)
    y = r * np.sin(angle)
    data_list.append([x, y])
    target_list.append([0, 0])
  for _ in range(num_outer):
    angle = np.random.uniform(0, 2 * np.pi)
    r = np.random.uniform(outer_radius - 0.15, outer_radius + 0.15)
    x = r * np.cos(angle)
    y = r * np.sin(angle)
    data_list.append([x, y])
    target_list.append([1, 1])
  data = np.array(data_list, dtype=np.float32)
  target = np.array(target_list, dtype=np.float32)
  w1, b1, w2, b2 = init_params_bilayer()
  for _ in range(3):
    gradient_descent_bilayer(data, target, w1, b1, w2, b2, sigmoid, dSigmoid)
  prediction = forward_bilayer(data, w1, b1, w2, b2, sigmoid)
  print(f"prediction (first 10): {prediction[:10]}")
  print(f"target (first 10): {target[:10]}")
  plot_bilayer_boundary(data, target, w1, b1, w2, b2, sigmoid, "CIRCLE_Bilayer")

if __name__ == "__main__":
  test_bilayer()
  test_numeric_bilayer()
  circle_gate_bilayer()
