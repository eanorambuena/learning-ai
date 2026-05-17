#!/home/eanorambuena/miniconda/envs/learning-ai/bin/python
from numba import njit, prange
import numpy as np
from utils import forward, sigmoid, dSigmoid, identity, dIdentity, mean_reduce
from test import test
from test_numeric import test_numeric

N: np.int32 = 2
LEARNING_RATE: np.float32 = 0.01
EPOCHS: np.int32 = 5000

@njit(parallel=True)
def compute_gradients(data: np.ndarray, target: np.ndarray, w: np.ndarray, b: np.ndarray, dw: np.ndarray, db: np.ndarray, activation, d_activation) -> None:
  SAMPLES, N = data.shape
  z = np.zeros(N, dtype=np.float32)
  a = np.zeros(N, dtype=np.float32)
  da_dz = np.zeros(N, dtype=np.float32)
  for i in prange(SAMPLES):
    for j in range(N):
      z[j] = b[j]
      for k in range(N):
        z[j] += w[j, k] * data[i, k]
    a = activation(z)
    da_dz = d_activation(z)
    for j in range(N):
      error = (a[j] - target[i, j]) * da_dz[j]
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
    dw_mean = mean_reduce(dw)
    db_mean = np.zeros(N, dtype=np.float32)
    for j in range(N):
      for i in range(SAMPLES):
        db_mean[j] += db[i, j]
      db_mean[j] /= SAMPLES
    w[:] = w - LEARNING_RATE * dw_mean
    b[:] = b - LEARNING_RATE * db_mean

if __name__ == "__main__":
  test(gradient_descent, forward, sigmoid, dSigmoid)
  test_numeric(gradient_descent, forward, identity, dIdentity)
