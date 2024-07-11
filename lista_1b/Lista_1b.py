import numpy as np
import matplotlib.pyplot as plt
import time

# Classe implementada para gerar o dataset
class DataGenerator:
    def __init__(self, num_points=100, seed=None):
        self.num_points = num_points
        if seed is not None:
            np.random.seed(seed)
        self.data = np.random.rand(num_points, 2)
        self.labels = np.zeros(num_points)
        self.slope, self.intercept = self._generate_random_line()
        self._classify_points()

    def _generate_random_line(self):
        slope = np.random.uniform(-1, 1)
        intercept = np.random.uniform(0, 1)
        return slope, intercept

    def _classify_points(self):
        for i, (x, y) in enumerate(self.data):
            if y > self.slope * x + self.intercept:
                self.labels[i] = 1
            else:
                self.labels[i] = 0

    def get_data_and_labels(self):
        return self.data, self.labels

    def plot_data(self):
        plt.figure(figsize=(8, 6))
        scatter = plt.scatter(self.data[:, 0], self.data[:, 1], c=self.labels, edgecolors='k', marker='o', cmap=plt.cm.coolwarm)
        x_vals = np.array([0, 1])
        y_vals = self.slope * x_vals + self.intercept
        plt.plot(x_vals, y_vals, color='black', linestyle='--', label='Função Geradora')
        plt.xlabel('$x_1$')
        plt.ylabel('$x_2$')
        plt.title('Conjunto de Dados')
        plt.legend(handles=scatter.legend_elements()[0] + [plt.Line2D([], [], color='black', linestyle='--')], labels=['Classe 0', 'Classe 1', 'Função Geradora'])
        plt.show()

# Classe implementada para criar e treinar o Perceptron e fazer classificações com ele
class Perceptron:
    def __init__(self, input_size, learning_rate=0.01, activation_function='sigmoid'):
        if activation_function == 'relu':
            self.weights = np.random.randn(input_size + 1) * np.sqrt(2 / input_size)  # He initialization
        else:
            self.weights = np.random.randn(input_size + 1)  # Initialize weights with small random numbers
        self.learning_rate = learning_rate
        self.activation_function = activation_function

    def activation(self, x):
        if self.activation_function == 'relu':
            return np.maximum(0, x)
        elif self.activation_function == 'sigmoid':
            return 1 / (1 + np.exp(-x))

    def activation_derivative(self, x):
        if self.activation_function == 'relu':
            return np.where(x > 0, 1, 0)
        elif self.activation_function == 'sigmoid':
            sigmoid_x = self.activation(x)
            return sigmoid_x * (1 - sigmoid_x)

    def predict(self, inputs):
        inputs_with_bias = np.append(inputs, 1)
        weighted_sum = np.dot(self.weights, inputs_with_bias)
        return self.activation(weighted_sum)

    def train(self, training_data, labels, epochs=100):
        for _ in range(epochs):
            for inputs, label in zip(training_data, labels):
                inputs_with_bias = np.append(inputs, 1)
                weighted_sum = np.dot(self.weights, inputs_with_bias)
                prediction = self.activation(weighted_sum)
                error = label - prediction
                derivative = self.activation_derivative(weighted_sum)
                update = self.learning_rate * error * derivative * inputs_with_bias
                self.weights += update
        return self.weights
    
    def plot_decision_boundary(self, training_data, labels):
        plt.figure(figsize=(8, 6))
        # Plot data points
        scatter = plt.scatter(training_data[:, 0], training_data[:, 1], c=labels, edgecolors='k', marker='o', cmap=plt.cm.coolwarm)
        
        # Calculate the decision boundary
        x_vals = np.array([0, 1])
        y_vals = -(self.weights[0] * x_vals + self.weights[2]) / self.weights[1]
        
        # Plot the decision boundary
        plt.plot(x_vals, y_vals, color='black', linestyle='--', label='Limite de Decisão')
        
        plt.xlabel('$x_1$')
        plt.ylabel('$x_2$')
        plt.title(f"Limite de Decisão ({self.activation_function})")
        plt.legend(handles=scatter.legend_elements()[0] + [plt.Line2D([], [], color='black', linestyle='--')], labels=['Classe 0', 'Classe 1', 'Limite de Decisão'])
        plt.show()

if __name__ == "__main__":
    data_gen = DataGenerator(num_points=200, seed=43)
    training_data, labels = data_gen.get_data_and_labels()
    
    data_gen.plot_data()

    epocas = 1000

    # Training with ReLU
    perceptron_relu = Perceptron(input_size=2, learning_rate=0.01, activation_function='relu')
    start_time_relu = time.time()
    weights_relu = perceptron_relu.train(training_data, labels, epochs=epocas)
    end_time_relu = time.time()
    time_elapsed_relu = end_time_relu - start_time_relu
    print(f"Pesos finais (ReLU): {weights_relu}")
    print(f"Tempo de treinamento (ReLU): {time_elapsed_relu:.5f} segundos")
    perceptron_relu.plot_decision_boundary(training_data, labels)
    
    # Training with Sigmoid
    perceptron_sigmoid = Perceptron(input_size=2, learning_rate=0.01, activation_function='sigmoid')
    start_time_sigmoid = time.time()
    weights_sigmoid = perceptron_sigmoid.train(training_data, labels, epochs=epocas)
    end_time_sigmoid = time.time()
    time_elapsed_sigmoid = end_time_sigmoid - start_time_sigmoid
    print(f"Pesos finais (Sigmoid): {weights_sigmoid}")
    print(f"Tempo de treinamento (Sigmoid): {time_elapsed_sigmoid:.5f} segundos")
    perceptron_sigmoid.plot_decision_boundary(training_data, labels)