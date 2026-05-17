# Run with: conda activate learning-ai && python 11_onehot_embedding.py
# Goal: Compress one-hot encoding (vocab_size -> embed_dim)

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# VOCABULARY - 10 words
# ============================================================
vocab = ["el", "la", "gato", "perro", "come", "duerme", "en", "casa", "el", "gato"]
vocab_size = len(vocab)
embed_dim = 2  # Compress to 2D for visualization

print(f"Vocab size: {vocab_size}")
print(f"Words: {vocab}")

# ============================================================
# Create one-hot encoding manually
# ============================================================
def one_hot(word_idx, vocab_size):
  vector = torch.zeros(vocab_size, dtype=torch.float32)
  vector[word_idx] = 1
  return vector

# Test one-hot
print(f"\nOne-hot for 'gato' (idx=2):")
print(one_hot(2, vocab_size))

# ============================================================
# Method 1: Using nn.Linear (manual approach)
# ============================================================
print("\n=== Method 1: nn.Linear ===")
linear_layer = nn.Linear(vocab_size, embed_dim, bias=False)

# Compress one-hot to embedding
gato_onehot = one_hot(2, vocab_size)
gato_embedding = linear_layer(gato_onehot)
print(f"Embedding for 'gato': {gato_embedding.detach().numpy()}")

# ============================================================
# Method 2: Using nn.Embedding (built-in)
# ============================================================
print("\n=== Method 2: nn.Embedding ===")
embedding_layer = nn.Embedding(vocab_size, embed_dim)

# Embedding layer expects index, not one-hot
gato_idx = torch.tensor([2])
gato_embedding2 = embedding_layer(gato_idx)
print(f"Embedding for 'gato': {gato_embedding2.detach().numpy()[0]}")

# ============================================================
# COMPARISON: Both do matrix multiplication!
# ============================================================
print("\n=== Compare weights ===")
print(f"Linear weight matrix:\n{linear_layer.weight.detach().numpy()}")
print(f"Embedding weight matrix:\n{embedding_layer.weight.detach().numpy()}")
print("These are the SAME concept - just different interfaces!")

# ============================================================
# VISUALIZE: Embeddings for all words
# ============================================================
print("\n=== Visualize all word embeddings ===")
fig, ax = plt.subplots(figsize=(8, 8))

for i, word in enumerate(vocab):
  idx = torch.tensor([i])
  emb = embedding_layer(idx).detach().numpy()[0]
  ax.scatter(emb[0], emb[1], s=200)
  ax.annotate(word, (emb[0], emb[1]), fontsize=12, ha='center')

ax.set_xlim(-2, 2)
ax.set_ylim(-2, 2)
ax.set_xlabel('Embedding dimension 1')
ax.set_ylabel('Embedding dimension 2')
ax.set_title('Word Embeddings (random initialization)')
ax.grid(True)
plt.savefig('plots/11_embeddings_random.png', dpi=100, bbox_inches='tight')
print("Saved: plots/11_embeddings_random.png")

# ============================================================
# TRAIN: Make similar words have similar embeddings
# ============================================================
# Simple task: "el" + "gato" → similar to "el" + "perro"
# We'll create fake pairs and train

print("\n=== Training embeddings ===")

# Create training pairs (fake context)
pairs = [
  (0, 2),   # el-gato
  (0, 3),   # el-perro
  (2, 4),   # gato-come
  (3, 4),   # perro-come
  (2, 5),   # gato-duerme
  (3, 5),   # perro-duerme
]

# Train embeddings to be close for related words
optimizer = optim.Adam(embedding_layer.parameters(), lr=0.1)
criterion = nn.MSELoss()

for epoch in range(500):
  total_loss = 0
  for word_idx, context_idx in pairs:
    word = torch.tensor([word_idx])
    context = torch.tensor([context_idx])

    word_emb = embedding_layer(word)
    context_emb = embedding_layer(context)

    # Try to make related words have similar embeddings
    loss = criterion(word_emb, context_emb * 0.5)  # Target: half of context
    total_loss += loss

  optimizer.zero_grad()
  total_loss.backward()
  optimizer.step()

  if epoch % 100 == 0:
    print(f"Epoch {epoch}, Loss: {total_loss.item():.4f}")

# ============================================================
# VISUALIZE: After training
# ============================================================
fig, ax = plt.subplots(figsize=(8, 8))

for i, word in enumerate(vocab):
  idx = torch.tensor([i])
  emb = embedding_layer(idx).detach().numpy()[0]
  ax.scatter(emb[0], emb[1], s=200)
  ax.annotate(word, (emb[0], emb[1]), fontsize=12, ha='center')

ax.set_xlabel('Embedding dimension 1')
ax.set_ylabel('Embedding dimension 2')
ax.set_title('Word Embeddings (after training)')
ax.grid(True)
plt.savefig('plots/11_embeddings_trained.png', dpi=100, bbox_inches='tight')
print("Saved: plots/11_embeddings_trained.png")

print("\n=== Done ===")