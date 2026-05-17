# Run with: conda activate tfenv && python 09_sklearn_xor.py
#
# Learning: If sklearn early stops too soon (tol=0.0001 default), try increasing lr.
# Default lr=0.1 was too slow, changed to lr=0.5 to converge properly.

import numpy as np
import sklearn as sk
import sklearn.neural_network as sknn

# XOR data
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32)
y = np.array([0, 1, 1, 0], dtype=np.float32)  # 1D for sklearn

print('XOR inputs:', X.flatten())
print('XOR targets:', y.flatten())
print('Implement Sklearn model below!')

nn = [2, 8, 1]  # Neurons in each layer
lr = 0.5
n_epochs = 10000

model = sknn.MLPRegressor(hidden_layer_sizes=(nn[1],),
                          activation='logistic',
                          solver='sgd',
                          learning_rate_init=lr,
                          max_iter=n_epochs,
                          verbose=True)

# Train the model
model.fit(X, y)

# Predictions
predictions = model.predict(X)
print('Predictions after training:', predictions)
