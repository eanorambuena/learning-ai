#!/home/eanorambuena/miniconda/envs/tfenv/bin/python
from numba import njit, prange
import numpy as np
from utils import apply_gradients
from test import test

LEARNING_RATE: np.float32 = 0.1
EPOCHS: np.int32 = 1000

@njit(parallel=True)
def compute_gradients(data: np.ndarray, target: np.ndarray, w: np.ndarray, b: np.ndarray, dw: np.ndarray, db: np.ndarray, activation, d_activation) -> None:
  SAMPLES, N = data.shape
  for i in prange(SAMPLES):
    for j in range(N):
      error = (b[j] + np.dot(w[j], data[i]) - target[i, j])
      for k in range(N):
        dw[i, j, k] = error * data[i, k]
      db[i, j] = error

@njit
def gradient_descent(data: np.ndarray, target: np.ndarray, w: np.ndarray, b: np.ndarray, activation, d_activation) -> None:
  SAMPLES, N = data.shape
  dw = np.zeros((SAMPLES, N, N), dtype=np.float32)
  db = np.zeros((SAMPLES, N), dtype=np.float32)
  for epoch in range(EPOCHS):
    compute_gradients(data, target, w, b, dw, db, activation, d_activation)
    apply_gradients(w, b, dw, db, LEARNING_RATE)

if __name__ == "__main__":
  test(gradient_descent)
