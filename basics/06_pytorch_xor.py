"""
Minimal PyTorch neural network - written by user to learn PyTorch.
This file is intentionally NOT auto-generated to maximize learning.
"""
import torch
import torch.nn as nn
import torch.optim as optim

# ============== SIMPLE NEURAL NETWORK ==============

class SimpleNet(nn.Module):
  """Neural network similar to our Numba implementations."""
  
  def __init__(self, input_size, hidden_size, output_size):
    super(SimpleNet, self).__init__()
    self.layer1 = nn.Linear(input_size, hidden_size)
    self.layer2 = nn.Linear(hidden_size, output_size)
    self.activation = nn.Sigmoid()
  
  def forward(self, x):
    x = self.layer1(x)
    x = self.activation(x)
    x = self.layer2(x)
    x = self.activation(x)
    return x


# ============== TRAINING FUNCTION ==============

def train_xor(lr=0.5, epochs=1000):
  """Train XOR gate - tu escribirás esta función."""
  
  # TODO: Define data and target tensors
  data = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=torch.float32)
  target = torch.tensor([[0], [1], [1], [0]], dtype=torch.float32)
  
  # TODO: Create model
  model = SimpleNet(input_size=2, hidden_size=8, output_size=1)
  
  # TODO: Define loss function (MSE)
  criterion = nn.MSELoss()
  
  # TODO: Define optimizer (SGD)
  optimizer = optim.SGD(model.parameters(), lr=lr)
  
  # TODO: Training loop
  for epoch in range(epochs):
    optimizer.zero_grad()     # Clear gradients
    output = model(data)      # Forward pass
    loss = criterion(output, target)  # Compute loss
    loss.backward()            # Backward pass (auto-differentiation!)
    optimizer.step()          # Update weights
  
    if epoch % 100 == 0:
      print(f'Epoch {epoch}, Loss: {loss.item():.4f}')
  
  # TODO: Return trained model
  return model


# ============== MAIN - RUN TRAINING ==============

if __name__ == "__main__":
  print("=== Entrena tu primera red en PyTorch ===")
  print("Descomenta el código en train_xor() y ejecútalo")
  print()
  
  # Quick test that PyTorch works:
  x = torch.tensor([1.0, 2.0])
  y = x * 2 + 1
  print(f"PyTorch test: {x} * 2 + 1 = {y}")
  
  print("\nAhora completa train_xor()!")
  model = train_xor(lr=0.1, epochs=10000)
  test_input = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=torch.float32)
  with torch.no_grad():
    predictions = model(test_input)
  print("\nPredicciones para XOR:")
  print(predictions)
  