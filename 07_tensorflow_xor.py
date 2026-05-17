# Run with: conda activate tfenv && python 07_tensorflow_xor.py

import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior() # Disable Keras

# XOR data
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32)
y = np.array([[0], [1], [1], [0]], dtype=np.float32)

# ----------------------------------------------------------------------
# YOUR CODE HERE - Learn TensorFlow by implementing:
# 1. Sequential model with Dense layers
# 2. Compile with optimizer and loss
# 3. Fit and train
# 4. Predict
# ----------------------------------------------------------------------

print('XOR inputs:', X.flatten())
print('XOR targets:', y.flatten())
print('Implement TensorFlow model below!')

iX = tf.placeholder('float', shape=[None, 2], name='inputs')
iY = tf.placeholder('float', shape=[None, 1], name='targets')

nn = [2, 8, 1]  # Neurons in each layer
lr = 0.1

# Layer 1
W1 = tf.Variable(tf.random_normal([nn[0], nn[1]]), name='W1')
b1 = tf.Variable(tf.random_normal([nn[1]]), name='b1')
l1 = tf.sigmoid(tf.add(tf.matmul(iX, W1), b1))

# Layer 2
W2 = tf.Variable(tf.random_normal([nn[1], nn[2]]), name='W2')
b2 = tf.Variable(tf.random_normal([nn[2]]), name='b2')

pY = tf.sigmoid(tf.add(tf.matmul(l1, W2), b2)) # Predicted output

# Loss and optimizer
loss = tf.losses.mean_squared_error(labels=iY, predictions=pY)
optimizer = tf.train.GradientDescentOptimizer(learning_rate=lr).minimize(loss)

n_epochs = 10000

with tf.Session() as sess:
  sess.run(tf.global_variables_initializer())
  
  for epoch in range(n_epochs):
    _, _loss, _py = sess.run([optimizer, loss, pY], feed_dict={iX: X, iY: y})
    if epoch % 1000 == 0:
      print(f'Epoch {epoch}, Loss: {_loss:.4f}')
  
  # Predictions
  predictions = sess.run(pY, feed_dict={iX: X})
  print('Predictions after training:', predictions.flatten())
