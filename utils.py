import numpy as np
from numba import njit, prange

N: np.int32 = 2
SAMPLES: np.int32 = 4

@njit
def rand_bin(shape: tuple) -> np.ndarray:
  return np.clip(np.random.rand(*shape), 0, 1).astype(np.float32)

@njit
def init_params() -> tuple:
  return (rand_bin((N, N)), rand_bin((N,)))

@njit(parallel=True)
def forward(data: np.ndarray, w: np.ndarray, b: np.ndarray) -> np.ndarray:
  z = np.zeros((SAMPLES, N), dtype=np.float32)
  for i in prange(SAMPLES):
    for j in range(N):
      z[i, j] = b[j]
      for k in range(N):
        z[i, j] += w[j, k] * data[i, k]
  return z

@njit
def mse(prediction: np.ndarray, target: np.ndarray) -> np.ndarray:
  return np.mean((prediction - target) ** 2)

def print_results(w: np.ndarray, b: np.ndarray, prediction: np.ndarray, target: np.ndarray, error: np.ndarray) -> None:
  print(f"w: {w}")
  print(f"b: {b}")
  print(f"prediction: {prediction}")
  print(f"target: {target}")
  precision = (1 - error) * 100
  print(f"Precision: {precision:.2f}%")