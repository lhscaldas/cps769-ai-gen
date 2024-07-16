import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, Input
import matplotlib.pyplot as plt

# Desabilitar GPU para treinamento (se necessário)
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# Desativar operações personalizadas oneDNN
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

class PathPredictor:
    def __init__(self, path_type='quadrado', epochs=500, num_repeat=40):
        self.path_type = path_type
        self.epochs = epochs
        self.model = self.build_model()
        self.data = self.generate_data(num_repeat)
        self.x_train, self.y_train = self.prepare_data()
        
    def build_model(self):
        model = models.Sequential([
            Input(shape=(1, 2)),
            layers.LSTM(50, activation='relu'),
            layers.Dense(2)
        ])
        model.compile(optimizer='adam', loss='mse')
        return model

    def generate_square_path(self):
        return np.array([
            [0.25, 0.25],
            [0.75, 0.25],
            [0.75, 0.75],
            [0.25, 0.75],
            [0.25, 0.25]
        ])

    def generate_spiral_square(self, turns=10, initial_step=0.05, step_increment=0.02):
            path = []
            x, y = 0.5, 0.5
            path.append([x, y])
            direction = 0
            step = initial_step
            for turn in range(1, turns + 1):
                for _ in range(turn):
                    if direction == 0:
                        x += step
                    elif direction == 1:
                        y += step
                    elif direction == 2:
                        x -= step
                    elif direction == 3:
                        y -= step
                    path.append([x, y])
                direction = (direction + 1) % 4
                if direction % 2 == 0:
                    step += step_increment  # Incrementa o passo após cada volta completa
            return np.array(path)

    def generate_data(self, num_repeat):
        if self.path_type == 'quadrado':
            return np.tile(self.generate_square_path(), (num_repeat, 1))
        elif self.path_type == 'espiral':
            return np.tile(self.generate_spiral_square(), (num_repeat, 1))
        else:
            raise ValueError("path_type must be 'quadrado' or 'espiral'")

    def prepare_data(self):
        data = self.data
        x_train = data[:-1].reshape(-1, 1, 2)
        y_train = data[1:].reshape(-1, 2)
        return x_train, y_train

    def train_model(self):
        self.model.fit(self.x_train, self.y_train, epochs=self.epochs, verbose=0)

    def plot_predictions(self, initial_points):
        plt.figure(figsize=(10, 6))
        
        for point in initial_points:
            current_input = np.array(point).reshape(1, 1, 2)
            predictions = [point]

            if self.path_type == 'quadrado':
                num_iter = 4
            elif self.path_type == 'espiral':
                num_iter = 30
            
            for _ in range(num_iter):
                next_prediction = self.model.predict(current_input)
                predictions.append(next_prediction.flatten())
                current_input = next_prediction.reshape(1, 1, 2)
            
            predictions = np.array(predictions)
            
            # Plotar o ponto inicial
            plt.scatter(point[0], point[1], marker='o', s=100, label=f'Ponto Inicial {point}')
            
            # Plotar as previsões
            plt.plot(predictions[:, 0], predictions[:, 1], label=f'Predito a partir de {point}', linestyle='dashed', linewidth=2)
        
        # Plotar o caminho original para referência
        if self.path_type == 'quadrado':
            plt.plot(self.generate_square_path()[:, 0], self.generate_square_path()[:, 1], color='gray', label='Caminho Original')
        elif self.path_type == 'espiral':
            plt.plot(self.generate_spiral_square()[:, 0], self.generate_spiral_square()[:, 1], color='gray', label='Caminho Original')
        plt.legend()
        plt.show()

def selecao(epochs=300, num_repeat=4):
    square_predictor = PathPredictor(path_type='quadrado', epochs=epochs, num_repeat=num_repeat)
    square_predictor.train_model()
    initial_points_square = [
        [0.25, 0.25],
    ]
    square_predictor.plot_predictions(initial_points_square)

def quadrado():
    square_predictor = PathPredictor(path_type='quadrado')
    square_predictor.train_model()
    initial_points_square = [
        [0.25, 0.25],
        [0.75, 0.25],
        [0.75, 0.75],
        [0.25, 0.75]
    ]
    square_predictor.plot_predictions(initial_points_square)

def espiral():
    spiral_predictor = PathPredictor(path_type='espiral')
    spiral_predictor.train_model()
    initial_points_spiral = [
        [0.5, 0.5],
        [0.34, 0.60],
        [0.79, 0.32],
        [0.79, 0.86]
    ]
    spiral_predictor.plot_predictions(initial_points_spiral)


if __name__ == "__main__":
    # Seleção de hiperparâmetros
    # selecao(epochs=100, num_repeat=40)

    # Quadrado
    # quadrado()

    # Espiral quadrada
    espiral()