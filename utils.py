import numpy as np
from numba import njit

@njit
def rand_bin(shape: tuple) -> np.ndarray:
  return np.clip(np.random.rand(*shape), 0, 1).astype(np.float32)

@njit
def sigmoid(x: np.ndarray) -> np.ndarray:
  return 1 / (1 + np.exp(-x))

@njit
def dSigmoid_dz(x: np.ndarray) -> np.ndarray:
  s = sigmoid(x)
  return s * (1 - s)

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
