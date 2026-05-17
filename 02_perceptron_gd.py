#!/home/eanorambuena/miniconda/envs/learning-ai/bin/python
from numba import njit, prange
import numpy as np
from utils import forward, identity, dIdentity, mean_reduce
from test import test

N: np.int32 = 2
SAMPLES: np.int32 = 4
LEARNING_RATE: np.float32 = 0.1
EPOCHS: np.int32 = 1000

@njit(parallel=True)
def compute_gradients(data: np.ndarray, target: np.ndarray, w: np.ndarray, b: np.ndarray, dw: np.ndarray, db: np.ndarray, activation, d_activation) -> None:
  for i in prange(SAMPLES):
    for j in range(N):
      error = (b[j] + np.dot(w[j], data[i]) - target[i, j])
      for k in range(N):
        dw[i, j, k] = error * data[i, k]
      db[i, j] = error

@njit
def gradient_descent(data: np.ndarray, target: np.ndarray, w: np.ndarray, b: np.ndarray, activation, d_activation) -> None:
  dw = np.zeros((SAMPLES, N, N), dtype=np.float32)
  db = np.zeros((SAMPLES, N), dtype=np.float32)
  for epoch in range(EPOCHS):
    compute_gradients(data, target, w, b, dw, db, activation, d_activation)
    dw_mean = mean_reduce(dw)
    db_mean = np.zeros(N, dtype=np.float32)
    for j in range(N):
      for i in range(SAMPLES):
        db_mean[j] += db[i, j]
      db_mean[j] /= SAMPLES
    w[:] = w - LEARNING_RATE * dw_mean
    b[:] = b - LEARNING_RATE * db_mean

if __name__ == "__main__":
  test(gradient_descent, forward)
