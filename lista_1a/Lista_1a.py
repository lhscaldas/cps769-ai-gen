import time
start_time = time.time()

# Modify the Python Script to Disable GPU
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# to run the python script: python3 rnn_cyclic_sequence.py

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt

# Define the square path coordinates
square_path = np.array([
    [0.25, 0.25],
    [0.75, 0.25],
    [0.75, 0.75],
    [0.25, 0.75],
    [0.25, 0.25]
])

# Generate the training data by repeating the square path
num_repeats = 40
data = np.tile(square_path, (num_repeats, 1))

# Initial point modification
mod_square_path = square_path
mod_square_path[0] = [0.25, 0.25]
mod_data = np.tile(mod_square_path, (num_repeats, 1))

# Prepare training data
x_train = mod_data[:-1].reshape(-1, 1, 2)
y_train = data[1:].reshape(-1, 2)

# Define the RNN model
model = models.Sequential([
    layers.LSTM(50, activation='relu', input_shape=(num_repeats, 2)),
    layers.Dense(2)
])

# Compile the model
model.compile(optimizer='adam', loss='mse')

# Train the model
model.fit(x_train, y_train, epochs=900, verbose=0)

fit_time = time.time()
print(f"Tempo de execução até o treinemento: {fit_time - start_time} segundos")

# Generate predictions
predictions = model.predict(x_train[:5])

predict_time = time.time()
print(f"Tempo de execução até a predição: {predict_time - start_time} segundos")

# Plot the results
plt.figure(figsize=(10, 6))
plt.plot(data[:, 0], data[:, 1], label='Original Path', linestyle='dashed', color='gray')
plt.plot(predictions[:, 0], predictions[:, 1], label='Predicted Path', color='blue')
plt.scatter(square_path[:, 0], square_path[:, 1], color='red')
plt.legend()
plt.show()