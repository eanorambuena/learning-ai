#!/home/eanorambuena/miniconda/envs/tfenv/bin/python
"""
Generic N-layer neural network implementation.
This file was generated with AI assistance.

Allows specifying network architecture via layer specification:
  layers = [[sigmoid, 4], [sigmoid, 2]]
  This creates: input → 4 neurons (sigmoid) → 2 neurons (sigmoid) → output
  Format: [[activation, num_neurons], [activation, num_neurons], ...]
"""
from numba import njit, prange
import numpy as np
from utils import sigmoid, dSigmoid, identity, dIdentity


def init_network(layer_sizes, input_size):
  """
  Initialize network with known input size.
  
  Args:
    layer_sizes: array of layer sizes
    input_size: number of input features
  """
  num_layers = len(layer_sizes)
  weights = []
  biases = []
  
  for i in range(num_layers):
    neurons = int(layer_sizes[i])
    if i == 0:
      weights.append(np.random.rand(neurons, input_size).astype(np.float32))
    else:
      prev_neurons = int(layer_sizes[i-1])
      weights.append(np.random.rand(neurons, prev_neurons).astype(np.float32))
    biases.append(np.random.rand(neurons).astype(np.float32))
  
  return weights, biases


@njit
def apply_activation_scalar(z, activation_type):
  """Apply activation by type: 0=sigmoid, 1=identity"""
  if activation_type == 0:
    return 1.0 / (1.0 + np.exp(-z))
  else:
    return z


@njit
def apply_d_activation_scalar(z, activation_type):
  """Apply derivative of activation by type"""
  if activation_type == 0:
    s = 1.0 / (1.0 + np.exp(-z))
    return s * (1.0 - s)
  else:
    return 1.0


@njit(parallel=True)
def forward_network(data, weights, biases, layer_sizes, activation_types):
  """
  Forward pass for N layers.
  
  Args:
    data: (SAMPLES, input_n)
    weights: list of weight matrices
    biases: list of bias vectors
    layer_sizes: array of layer sizes
    activation_types: array of activation types (0=sigmoid, 1=identity)
  """
  SAMPLES = data.shape[0]
  num_layers = len(weights)
  
  a_list = []
  z_list = []
  
  current = data.copy()
  a_list.append(current)
  
  for layer in range(num_layers):
    num_neurons = int(layer_sizes[layer])
    activation_type = int(activation_types[layer])
    z = np.zeros((SAMPLES, num_neurons), dtype=np.float32)
    a = np.zeros((SAMPLES, num_neurons), dtype=np.float32)
    
    for i in prange(SAMPLES):
      for j in range(num_neurons):
        z[i, j] = biases[layer][j]
        for k in range(weights[layer].shape[1]):
          z[i, j] += weights[layer][j, k] * current[i, k]
        a[i, j] = apply_activation_scalar(z[i, j], activation_type)
    
    z_list.append(z)
    a_list.append(a)
    current = a
  
  return a_list[-1], a_list, z_list


@njit
def mean_reduce_nd(arr):
  """
  Mean reduce for N-dimensional arrays.
  """
  SAMPLES = arr.shape[0]
  shape = arr.shape[1:]
  mean = np.zeros(shape, dtype=np.float32)
  
  for idx in np.ndindex(shape):
    total = 0.0
    for s in range(SAMPLES):
      total += arr[s][idx]
    mean[idx] = total / SAMPLES
  
  return mean


def train_network(layer_spec, data, target, lr=0.1, epochs=1000):
  """
  Train the generic network.
  
  Args:
    layer_spec: [[activation_type, num_neurons], ...]
      - activation_type: 0=sigmoid, 1=identity
      - num_neurons: number of neurons in this layer
    data: (SAMPLES, input_n)
    target: (SAMPLES, output_n)
    lr: learning rate
    epochs: number of epochs
  
  Returns:
    weights, biases
  """
  input_n = data.shape[1]
  output_n = target.shape[1]
  
  # Extract sizes and activation types
  layer_sizes = np.array([layer[1] for layer in layer_spec], dtype=np.int32)
  activation_types = np.array([layer[0] for layer in layer_spec], dtype=np.int32)
  
  # Adjust last layer to match output size
  layer_sizes[-1] = output_n
  
  weights, biases = init_network(layer_sizes, input_n)
  
  SAMPLES = data.shape[0]
  num_layers = len(weights)
  
  # Create gradient buffers
  dws = []
  dbs = []
  for i in range(num_layers):
    dws.append(np.zeros_like(weights[i]))
    dbs.append(np.zeros_like(biases[i]))
  
  for epoch in range(epochs):
    # Forward pass
    output, a_list, z_list = forward_network(data, weights, biases, layer_sizes, activation_types)
    
    # Initialize gradients
    for i in range(num_layers):
      dws[i].fill(0)
      dbs[i].fill(0)
    
    # Backprop
    for s in range(SAMPLES):
      # Error at output layer
      output_n_neurons = weights[-1].shape[0]
      error = np.zeros(output_n_neurons, dtype=np.float32)
      activation_type = int(activation_types[-1])
      
      for j in range(output_n_neurons):
        d_act = apply_d_activation_scalar(z_list[-1][s, j], activation_type)
        error[j] = (a_list[-1][s, j] - target[s, j]) * d_act
      
      # Propagate through layers
      for layer in range(num_layers - 1, -1, -1):
        prev_a = a_list[layer]
        prev_size = weights[layer].shape[1] if layer > 0 else input_n
        
        for j in range(weights[layer].shape[0]):
          dbs[layer][j] += error[j]
          for k in range(prev_size):
            dws[layer][j, k] += error[j] * prev_a[s, k]
        
        # Propagate error to previous layer
        if layer > 0:
          prev_error = np.zeros(weights[layer-1].shape[0], dtype=np.float32)
          for j in range(weights[layer].shape[0]):
            for k in range(weights[layer].shape[0]):
              prev_error[k] += weights[layer][j, k] * error[j]
          
          prev_activation_type = int(activation_types[layer-1])
          for k in range(len(prev_error)):
            d_act = apply_d_activation_scalar(z_list[layer-1][s, k], prev_activation_type)
            prev_error[k] *= d_act
          error = prev_error
    
    # Apply mean and gradients
    for i in range(num_layers):
      dw_mean = dws[i] / SAMPLES
      db_mean = dbs[i] / SAMPLES
      weights[i] -= lr * dw_mean
      biases[i] -= lr * db_mean
  
  return weights, biases


def plot_nlayer_boundary(data, target, weights, biases, layer_spec, title):
  """
  Plot decision boundary for N-layer network.
  """
  try:
    import matplotlib.pyplot as plt
    
    layer_sizes = np.array([layer[1] for layer in layer_spec], dtype=np.int32)
    activation_types = np.array([layer[0] for layer in layer_spec], dtype=np.int32)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    x_margin = (data[:, 0].max() - data[:, 0].min()) * 0.2
    y_margin = (data[:, 1].max() - data[:, 1].min()) * 0.2
    x_min, x_max = data[:, 0].min() - x_margin, data[:, 0].max() + x_margin
    y_min, y_max = data[:, 1].min() - y_margin, data[:, 1].max() + y_margin
    
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
    grid = np.c_[xx.ravel(), yy.ravel()].astype(np.float32)
    
    # Forward pass for grid
    output, _, _ = forward_network(grid, weights, biases, layer_sizes, activation_types)
    z_class = (output[:, 0] > 0.5).astype(float).reshape(xx.shape)
    
    ax.contourf(xx, yy, z_class, alpha=0.3, cmap='coolwarm')
    
    for i in range(data.shape[0]):
      t = target[i, 0] > 0.5
      ax.scatter(data[i, 0], data[i, 1], c='blue' if t else 'red', s=200, edgecolors='black')
    
    ax.set_xlabel('Input 1')
    ax.set_ylabel('Input 2')
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    
    import os
    os.makedirs("plots", exist_ok=True)
    plt.savefig(f"plots/{title.replace(' ', '_')}.png")
    plt.close()
  except Exception as e:
    print(f"Could not generate plot: {e}")


# ============== TESTS ==============
# Format: [[activation_type, num_neurons], ...]
# activation_type: 0=sigmoid, 1=identity

def test_monolayer():
  """Test monolayer (equivalent to 03)"""
  print("\n=== MONOLAYER TEST ===")
  layer_spec = [[0, 2], [0, 2]]  # sigmoid 2 neurons, sigmoid 2 neurons
  data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32)
  target = np.array([[0, 0], [1, 1], [1, 1], [0, 0]], dtype=np.float32)
  
  weights, biases = train_network(layer_spec, data, target, lr=0.5, epochs=3000)
  
  layer_sizes = np.array([layer[1] for layer in layer_spec], dtype=np.int32)
  activation_types = np.array([layer[0] for layer in layer_spec], dtype=np.int32)
  output, _, _ = forward_network(data, weights, biases, layer_sizes, activation_types)
  print(f"XOR prediction: {output}")
  print(f"XOR target: {target}")
  
  plot_nlayer_boundary(data, target, weights, biases, layer_spec, "MONOLAYER_XOR")


def test_bilayer():
  """Test bilayer (equivalent to 04)"""
  print("\n=== BILAYER TEST ===")
  layer_spec = [[0, 4], [0, 2]]  # sigmoid 4 neurons, sigmoid 2 neurons
  data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32)
  target = np.array([[0, 0], [1, 1], [1, 1], [0, 0]], dtype=np.float32)
  
  weights, biases = train_network(layer_spec, data, target, lr=0.5, epochs=3000)
  
  layer_sizes = np.array([layer[1] for layer in layer_spec], dtype=np.int32)
  activation_types = np.array([layer[0] for layer in layer_spec], dtype=np.int32)
  output, _, _ = forward_network(data, weights, biases, layer_sizes, activation_types)
  print(f"XOR prediction: {output}")
  print(f"XOR target: {target}")
  
  plot_nlayer_boundary(data, target, weights, biases, layer_spec, "BILAYER_XOR")


def test_trilayer():
  """Test trilayer (new!)"""
  print("\n=== TRILAYER TEST ===")
  layer_spec = [[0, 8], [0, 4], [0, 2]]  # sigmoid 8, sigmoid 4, sigmoid 2
  data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32)
  target = np.array([[0, 0], [1, 1], [1, 1], [0, 0]], dtype=np.float32)
  
  weights, biases = train_network(layer_spec, data, target, lr=0.5, epochs=3000)
  
  layer_sizes = np.array([layer[1] for layer in layer_spec], dtype=np.int32)
  activation_types = np.array([layer[0] for layer in layer_spec], dtype=np.int32)
  output, _, _ = forward_network(data, weights, biases, layer_sizes, activation_types)
  print(f"XOR prediction: {output}")
  print(f"XOR target: {target}")
  
  plot_nlayer_boundary(data, target, weights, biases, layer_spec, "TRILAYER_XOR")


def test_circle():
  """Test circle classification"""
  print("\n=== CIRCLE TEST (Bilayer) ===")
  layer_spec = [[0, 4], [0, 2]]  # sigmoid 4, sigmoid 2
  
  np.random.seed(42)
  inner_radius = 0.4
  outer_radius = 0.8
  num_inner = 30
  num_outer = 60
  
  data_list = []
  target_list = []
  
  for _ in range(num_inner):
    angle = np.random.uniform(0, 2 * np.pi)
    r = np.random.uniform(0, inner_radius)
    data_list.append([r * np.cos(angle), r * np.sin(angle)])
    target_list.append([0, 0])
  
  for _ in range(num_outer):
    angle = np.random.uniform(0, 2 * np.pi)
    r = np.random.uniform(outer_radius - 0.15, outer_radius + 0.15)
    data_list.append([r * np.cos(angle), r * np.sin(angle)])
    target_list.append([1, 1])
  
  data = np.array(data_list, dtype=np.float32)
  target = np.array(target_list, dtype=np.float32)
  
  # Train multiple times for better results
  weights, biases = None, None
  for _ in range(3):
    weights, biases = train_network(layer_spec, data, target, lr=0.5, epochs=3000)
  
  layer_sizes = np.array([layer[1] for layer in layer_spec], dtype=np.int32)
  activation_types = np.array([layer[0] for layer in layer_spec], dtype=np.int32)
  output, _, _ = forward_network(data, weights, biases, layer_sizes, activation_types)
  
  print(f"Inner predictions (first 5): {output[:5, 0]}")
  print(f"Outer predictions (first 5): {output[-5:, 0]}")
  
  plot_nlayer_boundary(data, target, weights, biases, layer_spec, "CIRCLE_Generic")


def test_add():
  """Test ADD with identity (regression)"""
  print("\n=== ADD TEST (Identity) ===")
  layer_spec = [[1, 4], [1, 2]]  # identity 4 neurons, identity 2 neurons
  
  data = np.array([
    [1, 2], [2, 3], [3, 1], [4, 2],
    [5, 3], [6, 1], [7, 4], [2, 5],
    [8, 2], [3, 6], [4, 4], [1, 7],
  ], dtype=np.float32)
  
  target = np.array([
    [3, -1], [5, -1], [4, 2], [6, 2],
    [8, 2], [7, 5], [11, 3], [7, -3],
    [10, 6], [9, -3], [8, 0], [8, -6],
  ], dtype=np.float32)
  
  weights, biases = train_network(layer_spec, data, target, lr=0.01, epochs=5000)
  
  layer_sizes = np.array([layer[1] for layer in layer_spec], dtype=np.int32)
  activation_types = np.array([layer[0] for layer in layer_spec], dtype=np.int32)
  output, _, _ = forward_network(data, weights, biases, layer_sizes, activation_types)
  print(f"ADD prediction: {output}")
  print(f"ADD target: {target}")
  
  plot_nlayer_boundary(data, target, weights, biases, layer_spec, "ADD_Generic")


if __name__ == "__main__":
  test_monolayer()
  test_bilayer()
  test_trilayer()
  test_circle()
  test_add()