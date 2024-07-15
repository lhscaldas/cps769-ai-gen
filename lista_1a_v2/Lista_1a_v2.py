import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt

# Desabilitar GPU para treinamento (se necessário)
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# Definir as coordenadas do caminho quadrado
square_path = np.array([
    [0.25, 0.25],
    [0.75, 0.25],
    [0.75, 0.75],
    [0.25, 0.75],
    [0.25, 0.25]
])

def generate_spiral_square(turns=3, points_per_turn=4):
    path = []
    step = 0.1
    x, y = 0.5, 0.5
    for turn in range(turns):
        path.append([x, y])
        x += step * (turn + 1)
        path.append([x, y])
        y += step * (turn + 1)
        path.append([x, y])
        x -= step * (turn + 1)
        path.append([x, y])
        y -= step * (turn + 1)
    return np.array(path)

spiral_path = generate_spiral_square(turns=3)

# Gerar os dados de treinamento repetindo o caminho quadrado
num_repeats = 40
data = np.tile(spiral_path, (num_repeats, 1))

# Preparar os dados de treinamento
x_train = data[:-1].reshape(-1, 1, 2)
y_train = data[1:].reshape(-1, 2)

# Definir o modelo RNN
model = models.Sequential([
    layers.LSTM(50, activation='relu', input_shape=(1, 2)),
    layers.Dense(2)
])

# Compilar o modelo
model.compile(optimizer='adam', loss='mse')

# Treinar o modelo
model.fit(x_train, y_train, epochs=500, verbose=0)

# Função para fazer previsões com pontos iniciais variados
def plot_predictions_with_varied_initial_points(model, initial_points):
    plt.figure(figsize=(10, 6))
    
    for point in initial_points:
        current_input = np.array(point).reshape(1, 1, 2)
        predictions = [point]
        
        for _ in range(4):
            next_prediction = model.predict(current_input)
            predictions.append(next_prediction.flatten())
            current_input = next_prediction.reshape(1, 1, 2)
        
        predictions = np.array(predictions)
        
        # Plotar o ponto inicial
        plt.scatter(point[0], point[1], marker='o', s=100, label=f'Ponto Inicial {point}')
        
        # Plotar as previsões
        plt.plot(predictions[:, 0], predictions[:, 1], label=f'Predito a partir de {point}', linestyle='dashed', linewidth=4)
    
    # Plotar o caminho original para referência
    plt.plot(spiral_path[:, 0], spiral_path[:, 1], label='Caminho Original', linestyle='dashed', color='gray')
    plt.legend()
    plt.show()

# Pontos iniciais para a previsão

initial_points = [
    [0.25, 0.25],
    [0.75, 0.25],
    [0.75, 0.75],
    [0.25, 0.75]
]

initial_points = [
    [0.50, 0.50],
]

# Plotar previsões com pontos iniciais variados
plot_predictions_with_varied_initial_points(model, initial_points)