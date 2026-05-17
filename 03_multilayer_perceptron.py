#!/home/eanorambuena/miniconda/envs/learning-ai/bin/python
from numba import njit, prange
import numpy as np

N: np.int32 = 2
SAMPLES: np.int32 = 4
LEARNING_RATE: np.float32 = 0.1
EPOCHS: np.int32 = 1000

"""
Rows are samples, columns are features. eg:

    | 0 0 |              | 0 0 |
X = | 0 1 |     target = | 1 1 |
    | 1 0 |              | 1 1 |
    | 1 1 |              | 1 1 |

OR:   target = [[0,0], [1,1], [1,1], [1,1]]
AND:  target = [[0,0], [0,0], [0,0], [1,1]]
XOR:  target = [[0,0], [1,1], [1,1], [0,0]]
"""

@njit
def rand_bin(shape: tuple) -> np.ndarray:
  return np.clip(np.random.rand(*shape), 0, 1).astype(np.float32)

@njit
def sigmoid(x: np.ndarray) -> np.ndarray:
  return 1 / (1 + np.exp(-x))

@njit
def dSigmoid_dz(z: np.ndarray) -> np.ndarray:
  s = sigmoid(z)
  return s * (1 - s)

@njit
def mse(prediction: np.ndarray, target: np.ndarray) -> np.ndarray:
  return np.mean((prediction - target) ** 2)

@njit
def init_params() -> tuple:
  return (rand_bin((N, N)), rand_bin((N,)))

@njit(parallel=True)
def compute_gradients(data: np.ndarray, target: np.ndarray, w: np.ndarray, b: np.ndarray, dw: np.ndarray, db: np.ndarray, use_activation: bool) -> None:
  z = np.zeros(N, dtype=np.float32)
  a = np.zeros(N, dtype=np.float32)
  for i in prange(SAMPLES):
    for j in range(N):
      z[j] = b[j]
      for k in range(N):
        z[j] += w[j, k] * data[i, k]
    if use_activation:
      for j in range(N):
        a[j] = 1 / (1 + np.exp(-z[j]))
        da_dz = a[j] * (1 - a[j])
        error = (a[j] - target[i, j]) * da_dz
    else:
      for j in range(N):
        error = z[j] - target[i, j]
    for j in range(N):
      for k in range(N):
        dw[i, j, k] = error * data[i, k]
      db[i, j] = error

@njit
def gradient_descent(data: np.ndarray, target: np.ndarray, w: np.ndarray, b: np.ndarray, use_activation: bool) -> None:
  dw = np.zeros((SAMPLES, N, N), dtype=np.float32)
  db = np.zeros((SAMPLES, N), dtype=np.float32)
  dw_mean = np.zeros((N, N), dtype=np.float32)
  db_mean = np.zeros(N, dtype=np.float32)
  for epoch in range(EPOCHS):
    compute_gradients(data, target, w, b, dw, db, use_activation)
    for j in range(N):
      for k in range(N):
        dw_mean[j, k] = 0.0
        for i in range(SAMPLES):
          dw_mean[j, k] += dw[i, j, k]
        dw_mean[j, k] /= SAMPLES
      db_mean[j] = 0.0
      for i in range(SAMPLES):
        db_mean[j] += db[i, j]
      db_mean[j] /= SAMPLES
    w[:] = w - LEARNING_RATE * dw_mean
    b[:] = b - LEARNING_RATE * db_mean

@njit(parallel=True)
def forward(data: np.ndarray, w: np.ndarray, b: np.ndarray, use_activation: bool) -> np.ndarray:
  z = np.zeros((SAMPLES, N), dtype=np.float32)
  a = np.zeros((SAMPLES, N), dtype=np.float32)
  for i in prange(SAMPLES):
    for j in range(N):
      z[i, j] = b[j]
      for k in range(N):
        z[i, j] += w[j, k] * data[i, k]
    if use_activation:
      for j in range(N):
        a[i, j] = 1 / (1 + np.exp(-z[i, j]))
  if use_activation:
    return a
  return z

def init_and_train(data: np.ndarray, target: np.ndarray, use_activation: bool) -> None:
  w, b = init_params()
  gradient_descent(data, target, w, b, use_activation)
  prediction = forward(data, w, b, use_activation)
  print(f"w: {w}")
  print(f"b: {b}")
  print(f"prediction: {prediction}")
  print(f"target: {target}")
  error = mse(prediction, target)
  precision = (1 - error) * 100
  print(f"Precision: {precision:.2f}%")

def or_gate(use_activation: bool):
  print("\n=== OR ===")
  data: np.ndarray = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32)
  target: np.ndarray = np.array([[0, 0], [1, 1], [1, 1], [1, 1]], dtype=np.float32)
  init_and_train(data, target, use_activation)

def and_gate(use_activation: bool):
  print("\n=== AND ===")
  data: np.ndarray = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32)
  target: np.ndarray = np.array([[0, 0], [0, 0], [0, 0], [1, 1]], dtype=np.float32)
  init_and_train(data, target, use_activation)

def xor_gate(use_activation: bool):
  print("\n=== XOR ===")
  data: np.ndarray = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32)
  target: np.ndarray = np.array([[0, 0], [1, 1], [1, 1], [0, 0]], dtype=np.float32)
  init_and_train(data, target, use_activation)

def main():
  or_gate(True)
  and_gate(True)
  xor_gate(True)

if __name__ == "__main__":
  main()
