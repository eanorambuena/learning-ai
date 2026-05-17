#!/home/eanorambuena/miniconda/envs/learning-ai/bin/python
from numba import njit, prange
import numpy as np
from utils import init_params, forward, mse, print_results

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

@njit(parallel=True)
def compute_gradients(data: np.ndarray, target: np.ndarray, w: np.ndarray, b: np.ndarray, dw: np.ndarray, db: np.ndarray) -> None:
  for i in prange(SAMPLES):
    for j in range(N):
      error = (b[j] + np.dot(w[j], data[i]) - target[i, j])
      for k in range(N):
        dw[i, j, k] = error * data[i, k]
      db[i, j] = error

@njit
def gradient_descent(data: np.ndarray, target: np.ndarray, w: np.ndarray, b: np.ndarray) -> None:
  dw = np.zeros((SAMPLES, N, N), dtype=np.float32)
  db = np.zeros((SAMPLES, N), dtype=np.float32)
  dw_mean = np.zeros((N, N), dtype=np.float32)
  db_mean = np.zeros(N, dtype=np.float32)
  for epoch in range(EPOCHS):
    compute_gradients(data, target, w, b, dw, db)
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

def init_and_train(data: np.ndarray, target: np.ndarray) -> None:
  w, b = init_params()
  
  gradient_descent(data, target, w, b)

  prediction = forward(data, w, b)
  error = mse(prediction, target)
  print_results(w, b, prediction, target, error)

def or_gate():
  print("\n=== OR ===")
  data: np.ndarray = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32)
  target: np.ndarray = np.array([[0, 0], [1, 1], [1, 1], [1, 1]], dtype=np.float32)
  init_and_train(data, target)

def and_gate():
  print("\n=== AND ===")
  data: np.ndarray = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32)
  target: np.ndarray = np.array([[0, 0], [0, 0], [0, 0], [1, 1]], dtype=np.float32)
  init_and_train(data, target)

def xor_gate():
  print("\n=== XOR ===")
  data: np.ndarray = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32)
  target: np.ndarray = np.array([[0, 0], [1, 1], [1, 1], [0, 0]], dtype=np.float32)
  init_and_train(data, target)

def main():
  or_gate()
  and_gate()
  xor_gate()

if __name__ == "__main__":
  main()
