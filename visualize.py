"""
Visualization utilities for neural network decision boundaries.
This file was generated with AI assistance to create plots showing
the classification boundary for each logic gate.

The plots show:
- A 2D mesh grid covering the input space
- Decision regions colored by predicted class
- Training data points overlaid with target labels (blue=1, red=0)
"""
import numpy as np
import matplotlib.pyplot as plt
import os

os.makedirs("plots", exist_ok=True)


def plot_decision_boundary(data: np.ndarray, target: np.ndarray, w: np.ndarray, b: np.ndarray, forward_func, activation, title: str) -> None:
  """
  Plot decision boundary for a monolayer neural network.
  
  Args:
    data: Input data (SAMPLES, 2)
    target: Target values (SAMPLES, 2)
    w: Weights matrix
    b: Bias vector
    forward_func: Forward pass function
    activation: Activation function
    title: Plot title
  """
  fig, ax = plt.subplots(figsize=(8, 6))
  x_margin = (data[:, 0].max() - data[:, 0].min()) * 0.2
  y_margin = (data[:, 1].max() - data[:, 1].min()) * 0.2
  x_min, x_max = data[:, 0].min() - x_margin, data[:, 0].max() + x_margin
  y_min, y_max = data[:, 1].min() - y_margin, data[:, 1].max() + y_margin
  xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
  grid = np.c_[xx.ravel(), yy.ravel()].astype(np.float32)
  z = forward_func(grid, w, b, activation)
  z_class = (z[:, 0] > 0.5).astype(float).reshape(xx.shape)
  ax.contourf(xx, yy, z_class, alpha=0.3, cmap='coolwarm')
  for i in range(data.shape[0]):
    t = target[i, 0] > 0.5
    ax.scatter(data[i, 0], data[i, 1], c='blue' if t else 'red', s=200, edgecolors='black', marker='o')
  ax.set_xlabel('Input 1')
  ax.set_ylabel('Input 2')
  ax.set_title(title)
  ax.set_xlim(x_min, x_max)
  ax.set_ylim(y_min, y_max)
  ax.grid(True, alpha=0.3)
  plt.savefig(f"plots/{title.replace(' ', '_')}.png")
  plt.close()


def plot_bilayer_boundary(data: np.ndarray, target: np.ndarray, w1, b1, w2, b2, activation, title: str) -> None:
  """
  Plot decision boundary for a bilayer neural network.
  
  Args:
    data: Input data (SAMPLES, 2)
    target: Target values (SAMPLES, 2)
    w1, b1: First layer weights and bias
    w2, b2: Second layer weights and bias
    activation: Activation function
    title: Plot title
  """
  HIDDEN_N = w1.shape[0]
  OUTPUT_N = w2.shape[0]
  fig, ax = plt.subplots(figsize=(8, 6))
  x_margin = (data[:, 0].max() - data[:, 0].min()) * 0.2
  y_margin = (data[:, 1].max() - data[:, 1].min()) * 0.2
  x_min, x_max = data[:, 0].min() - x_margin, data[:, 0].max() + x_margin
  y_min, y_max = data[:, 1].min() - y_margin, data[:, 1].max() + y_margin
  xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
  grid = np.c_[xx.ravel(), yy.ravel()].astype(np.float32)
  hidden = np.zeros((grid.shape[0], HIDDEN_N), dtype=np.float32)
  for i in range(grid.shape[0]):
    for j in range(HIDDEN_N):
      hidden[i, j] = b1[j]
      for k in range(2):
        hidden[i, j] += w1[j, k] * grid[i, k]
    hidden[i] = activation(hidden[i])
  output = np.zeros((grid.shape[0], OUTPUT_N), dtype=np.float32)
  for i in range(grid.shape[0]):
    for j in range(OUTPUT_N):
      output[i, j] = b2[j]
      for k in range(HIDDEN_N):
        output[i, j] += w2[j, k] * hidden[i, k]
    output[i] = activation(output[i])
  z_class = (output[:, 0] > 0.5).astype(float).reshape(xx.shape)
  ax.contourf(xx, yy, z_class, alpha=0.3, cmap='coolwarm')
  for i in range(data.shape[0]):
    t = target[i, 0] > 0.5
    ax.scatter(data[i, 0], data[i, 1], c='blue' if t else 'red', s=200, edgecolors='black', marker='o')
  ax.set_xlabel('Input 1')
  ax.set_ylabel('Input 2')
  ax.set_title(title)
  ax.set_xlim(x_min, x_max)
  ax.set_ylim(y_min, y_max)
  ax.grid(True, alpha=0.3)
  plt.savefig(f"plots/{title.replace(' ', '_')}.png")
  plt.close()