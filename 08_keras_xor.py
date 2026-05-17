# Run with: conda activate tfenv && python 08_keras_xor.py

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import tensorflow as tf
tf.random.set_seed(42)
np.random.seed(42)
from tensorflow import keras
from tensorflow.keras import layers

# XOR data
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32)
y = np.array([[0], [1], [1], [0]], dtype=np.float32)

print('XOR inputs:', X.flatten())
print('XOR targets:', y.flatten())

nn = [2, 8, 1]
lr = 0.1
n_epochs = 10000

# Manual Keras implementation (like 07_tensorflow)
inputs = keras.Input(shape=(2,), name='inputs')
x = layers.Dense(nn[1], activation='sigmoid', name='hidden')(inputs)
outputs = layers.Dense(nn[2], activation='sigmoid', name='output')(inputs)

model = keras.Model(inputs=inputs, outputs=outputs)
model.compile(optimizer=keras.optimizers.SGD(learning_rate=lr), loss='mse')

# Train with fit (batch_size=4 means full batch like TF1)
history = model.fit(X, y, epochs=n_epochs, batch_size=4, verbose=0)
for epoch in range(0, n_epochs, 1000):
  print(f'Epoch {epoch}, Loss: {history.history["loss"][epoch // 1000]:.4f}')

predictions = model.predict(X, verbose=0)
print('Predictions after training:', predictions.flatten())