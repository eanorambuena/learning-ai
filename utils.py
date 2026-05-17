import numpy as np
from numba import njit, prange

@njit
def rand_bin(shape: tuple) -> np.ndarray:
  return np.clip(np.random.rand(*shape), 0, 1).astype(np.float32)

@njit
def init_params(n: np.int32) -> tuple:
  return (rand_bin((n, n)), rand_bin((n,)))

@njit
def identity(x: np.ndarray) -> np.ndarray:
  return x

@njit
def sigmoid(x: np.ndarray) -> np.ndarray:
  return 1 / (1 + np.exp(-x))

@njit
def dIdentity(x: np.ndarray) -> np.ndarray:
  return np.ones_like(x)

@njit
def dSigmoid(x: np.ndarray) -> np.ndarray:
  s = sigmoid(x)
  return s * (1 - s)

@njit(parallel=True)
def linear_forward(data: np.ndarray, w: np.ndarray, b: np.ndarray) -> np.ndarray:
  SAMPLES, N = data.shape
  z = np.zeros((SAMPLES, N), dtype=np.float32)
  for i in prange(SAMPLES):
    for j in range(N):
      z[i, j] = b[j]
      for k in range(N):
        z[i, j] += w[j, k] * data[i, k]
  return z

@njit(parallel=True)
def mean_reduce(arr: np.ndarray) -> np.ndarray:
  SAMPLES, N, _ = arr.shape
  mean = np.zeros((N, N), dtype=np.float32)
  for j in prange(N):
    for k in range(N):
      total = 0.0
      for i in range(SAMPLES):
        total += arr[i, j, k]
      mean[j, k] = total / SAMPLES
  return mean

@njit(parallel=True)
def mean_reduce_1d(arr: np.ndarray) -> np.ndarray:
  SAMPLES, N = arr.shape
  mean = np.zeros(N, dtype=np.float32)
  for j in prange(N):
    total = 0.0
    for i in range(SAMPLES):
      total += arr[i, j]
    mean[j] = total / SAMPLES
  return mean

@njit
def apply_gradients(w: np.ndarray, b: np.ndarray, dw: np.ndarray, db: np.ndarray, learning_rate: np.float32) -> None:
  dw_mean = mean_reduce(dw)
  db_mean = mean_reduce_1d(db)
  w[:] = w - learning_rate * dw_mean
  b[:] = b - learning_rate * db_mean

@njit(parallel=True)
def forward(data: np.ndarray, w: np.ndarray, b: np.ndarray, activation) -> np.ndarray:
  SAMPLES, N = data.shape
  z = linear_forward(data, w, b)
  a = np.zeros((SAMPLES, N), dtype=np.float32)
  for i in prange(SAMPLES):
    a[i] = activation(z[i])
  return a

@njit
def mse(prediction: np.ndarray, target: np.ndarray) -> np.ndarray:
  return np.mean((prediction - target) ** 2)

def print_results(w: np.ndarray, b: np.ndarray, prediction: np.ndarray, target: np.ndarray) -> None:
  error = mse(prediction, target)
  print(f"w: {w}")
  print(f"b: {b}")
  print(f"prediction: {prediction}")
  print(f"target: {target}")
  precision = (1 - error) * 100
  print(f"Precision: {precision:.2f}%")

def init_and_train(data: np.ndarray, target: np.ndarray, train_func, activation, d_activation) -> None:
  n = data.shape[1]
  w, b = init_params(n)
  train_func(data, target, w, b, activation, d_activation)
  prediction = forward(data, w, b, activation)
  print_results(w, b, prediction, target)
