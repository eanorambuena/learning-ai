#!/home/eanorambuena/miniconda/envs/tfenv/bin/python
from numba import njit, prange
import numpy as np
from utils import sigmoid, dSigmoid, apply_gradients
from test import test
from test_numeric import test_numeric

N: np.int32 = 2
LEARNING_RATE: np.float32 = 0.01
EPOCHS: np.int32 = 5000

@njit
def linear_forward_sample(data_row: np.ndarray, w: np.ndarray, b: np.ndarray) -> np.ndarray:
  """Calcula z = w @ input + b para una sola muestra."""
  N = w.shape[0]
  z = np.zeros(N, dtype=np.float32)
  for j in range(N):
    z[j] = b[j]
    for k in range(N):
      z[j] += w[j, k] * data_row[k]
  return z


@njit
def compute_output_error(a: np.ndarray, target_row: np.ndarray, da_dz: np.ndarray) -> np.ndarray:
  """Calcula error = (a - target) * da/dz."""
  N = a.shape[0]
  error = np.zeros(N, dtype=np.float32)
  for j in range(N):
    error[j] = (a[j] - target_row[j]) * da_dz[j]
  return error


@njit
def compute_gradients_sample(error: np.ndarray, data_row: np.ndarray, dw_sample: np.ndarray, db_sample: np.ndarray) -> None:
  """Calcula dw y db para una muestra."""
  N = error.shape[0]
  for j in range(N):
    db_sample[j] = error[j]
    for k in range(N):
      dw_sample[j, k] = error[j] * data_row[k]


@njit(parallel=True)
def compute_gradients(data: np.ndarray, target: np.ndarray, w: np.ndarray, b: np.ndarray, dw: np.ndarray, db: np.ndarray, activation, d_activation) -> None:
  SAMPLES, N = data.shape
  z = np.zeros(N, dtype=np.float32)
  a = np.zeros(N, dtype=np.float32)
  da_dz = np.zeros(N, dtype=np.float32)
  for i in prange(SAMPLES):
    z = linear_forward_sample(data[i], w, b)
    a = activation(z)
    da_dz = d_activation(z)
    error = compute_output_error(a, target[i], da_dz)
    compute_gradients_sample(error, data[i], dw[i], db[i])

@njit
def gradient_descent(data: np.ndarray, target: np.ndarray, w: np.ndarray, b: np.ndarray, activation, d_activation) -> None:
  SAMPLES, N = data.shape
  dw = np.zeros((SAMPLES, N, N), dtype=np.float32)
  db = np.zeros((SAMPLES, N), dtype=np.float32)
  for epoch in range(EPOCHS):
    compute_gradients(data, target, w, b, dw, db, activation, d_activation)
    apply_gradients(w, b, dw, db, LEARNING_RATE)

if __name__ == "__main__":
  test(gradient_descent, sigmoid, dSigmoid)
  test_numeric(gradient_descent)
