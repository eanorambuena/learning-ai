# Compare: PyTorch, TensorFlow, Keras, sklearn on CIRCLE classification
# ============================================================
# GLOBAL CONFIG - Change these values to tune all frameworks
# ============================================================
EPOCHS = 2000
LR = 0.5
HIDDEN_SIZE = 16
LOG_EVERY = 500

# ============================================================

import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Generate CIRCLE data
np.random.seed(42)
inner_radius = 0.4
outer_radius = 0.8
num_inner, num_outer = 30, 60

def generate_circle_data():
  data_list, target_list = [], []
  for _ in range(num_inner):
    angle = np.random.uniform(0, 2 * np.pi)
    r = np.random.uniform(0, inner_radius)
    data_list.append([r * np.cos(angle), r * np.sin(angle)])
    target_list.append([0])
  for _ in range(num_outer):
    angle = np.random.uniform(0, 2 * np.pi)
    r = np.random.uniform(outer_radius - 0.15, outer_radius + 0.15)
    data_list.append([r * np.cos(angle), r * np.sin(angle)])
    target_list.append([1])
  return np.array(data_list, dtype=np.float32), np.array(target_list, dtype=np.float32)

X, y = generate_circle_data()
print(f"CIRCLE: {len(X)} samples, {num_inner} inner, {num_outer} outer")

# ============================================================
# 1. PyTorch
# ============================================================
print("\n=== PyTorch ===")
try:
  import torch
  import torch.nn as nn
  import torch.optim as optim

  class CircleNet(nn.Module):
    def __init__(self):
      super().__init__()
      self.fc1 = nn.Linear(2, HIDDEN_SIZE)
      self.fc2 = nn.Linear(HIDDEN_SIZE, 1)
    def forward(self, x):
      x = torch.sigmoid(self.fc1(x))
      x = torch.sigmoid(self.fc2(x))
      return x

  model = CircleNet()
  optimizer = optim.SGD(model.parameters(), lr=LR)
  criterion = nn.MSELoss()

  X_t = torch.tensor(X)
  y_t = torch.tensor(y)

  for epoch in range(EPOCHS):
    optimizer.zero_grad()
    out = model(X_t)
    loss = criterion(out, y_t)
    loss.backward()
    optimizer.step()
    if epoch % LOG_EVERY == 0:
      preds = (model(X_t).detach().numpy() > 0.5).astype(int)
      acc = (preds.flatten() == y.flatten()).mean()
      print(f"Epoch {epoch}, Loss: {loss.item():.4f}, Accuracy: {acc:.2%}")
except Exception as e:
  print(f"PyTorch error: {e}")

# ============================================================
# 2. Keras (using tf.keras - no compat.v1)
# ============================================================
print("\n=== Keras ===")
try:
  import tensorflow as tf
  tf.get_logger().setLevel('ERROR')
  
  model = tf.keras.Sequential([
    tf.keras.layers.Dense(HIDDEN_SIZE, activation='sigmoid', input_shape=(2,)),
    tf.keras.layers.Dense(1, activation='sigmoid')
  ])
  model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=LR), loss='mse')

  for epoch in range(EPOCHS):
    model.train_on_batch(X, y)
  
  preds = model.predict(X, verbose=0) > 0.5
  acc = (preds.flatten() == y.flatten()).mean()
  print(f"Final Accuracy: {acc:.2%}")
except Exception as e:
  print(f"Keras error: {e}")

# ============================================================
# 3. TensorFlow (graph mode - legacy)
# ============================================================
print("\n=== TensorFlow (legacy) ===")
try:
  import tensorflow.compat.v1 as tf
  tf.disable_v2_behavior()

  iX = tf.placeholder(tf.float32, shape=[None, 2])
  iY = tf.placeholder(tf.float32, shape=[None, 1])

  W1 = tf.Variable(tf.random_normal([2, HIDDEN_SIZE]))
  b1 = tf.Variable(tf.random_normal([HIDDEN_SIZE]))
  l1 = tf.sigmoid(tf.matmul(iX, W1) + b1)

  W2 = tf.Variable(tf.random_normal([HIDDEN_SIZE, 1]))
  b2 = tf.Variable(tf.random_normal([1]))
  pY = tf.sigmoid(tf.matmul(l1, W2) + b2)

  loss = tf.losses.mean_squared_error(iY, pY)
  optimizer = tf.train.GradientDescentOptimizer(LR).minimize(loss)

  with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for epoch in range(EPOCHS):
      _, l = sess.run([optimizer, loss], feed_dict={iX: X, iY: y})
      if epoch % LOG_EVERY == 0:
        preds = sess.run(pY, feed_dict={iX: X}) > 0.5
        acc = (preds.flatten() == y.flatten()).mean()
        print(f"Epoch {epoch}, Loss: {l:.4f}, Accuracy: {acc:.2%}")
except Exception as e:
  print(f"TensorFlow error: {e}")
except Exception as e:
  print(f"Keras error: {e}")

# ============================================================
# 4. sklearn
# ============================================================
print("\n=== sklearn ===")
try:
  import sklearn.neural_network as sknn

  model = sknn.MLPRegressor(
    hidden_layer_sizes=(HIDDEN_SIZE,),
    activation='logistic',
    solver='adam',
    learning_rate_init=LR,
    max_iter=EPOCHS,
    early_stopping=False,
    verbose=False
  )
  model.fit(X, y.ravel())
  preds = model.predict(X) > 0.5
  acc = (preds == y.ravel()).mean()
  print(f"Final Accuracy: {acc:.2%}")
except Exception as e:
  print(f"sklearn error: {e}")

print("\n=== Done ===")
