#!/home/eanorambuena/miniconda/envs/learning-ai/bin/python
from numba import njit, prange
import numpy as np
from utils import forward
from test import test

N: np.int32 = 2
SAMPLES: np.int32 = 4

@njit(parallel=True)
def least_squares(data: np.ndarray, target: np.ndarray, w: np.ndarray, b: np.ndarray) -> None:
  XtX = data.T @ data
  XtX_inv = np.linalg.inv(XtX)
  XtY = data.T @ target
  w[:] = XtX_inv @ XtY
  residuals = target - data @ w
  for j in prange(N):
    b[j] = np.mean(residuals[:, j])

if __name__ == "__main__":
  test(least_squares, forward)
