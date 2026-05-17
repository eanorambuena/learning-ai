#!/home/eanorambuena/miniconda/envs/learning-ai/bin/python
from numba import njit, prange
import numpy as np

N: np.int32 = 2
SAMPLES: np.int32 = 4

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
def mse(prediction: np.ndarray, target: np.ndarray) -> np.ndarray:
  return np.mean((prediction - target) ** 2)

@njit
def init_params() -> tuple:
  return (rand_bin((N, N)), rand_bin((N,)))

@njit(parallel=True)
def least_squares(data: np.ndarray, target: np.ndarray, w: np.ndarray, b: np.ndarray) -> None:
  XtX = data.T @ data
  XtX_inv = np.linalg.inv(XtX)
  XtY = data.T @ target
  w[:] = XtX_inv @ XtY
  residuals = target - data @ w
  for j in prange(N):
    b[j] = np.mean(residuals[:, j])

@njit(parallel=True)
def forward(data: np.ndarray, w: np.ndarray, b: np.ndarray) -> np.ndarray:
  z = np.zeros((SAMPLES, N), dtype=np.float32)
  for i in prange(SAMPLES):
    for j in range(N):
      z[i, j] = b[j]
      for k in range(N):
        z[i, j] += w[j, k] * data[i, k]
  return z

def init_and_train(data: np.ndarray, target: np.ndarray) -> None:
  w, b = init_params()
  least_squares(data, target, w, b)
  prediction = forward(data, w, b)
  print(f"w: {w}")
  print(f"b: {b}")
  print(f"prediction: {prediction}")
  print(f"target: {target}")
  error = mse(prediction, target)
  precision = (1 - error) * 100
  print(f"Precision: {precision:.2f}%")

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
