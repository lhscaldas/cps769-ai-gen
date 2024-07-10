import numpy as np
import matplotlib.pyplot as plt

class DataGenerator:
    def __init__(self, num_points=100):
        self.num_points = num_points
        np.random.seed(43) # fixando a seed para o experimento ser reproduzível
        self.data = np.random.rand(num_points, 2)
        self.labels = np.zeros(num_points)
        self.slope, self.intercept = self._generate_random_line()
        self._classify_points()

    def _generate_random_line(self):
        # Generate a random slope between -1 and 1
        slope = np.random.uniform(-1, 1)
        # Generate a random intercept between 0 and 1
        intercept = np.random.uniform(0, 1)
        return slope, intercept

    def _classify_points(self):
        for i, (x, y) in enumerate(self.data):
            # Classify points based on the line y = slope * x + intercept
            if y > self.slope * x + self.intercept:
                self.labels[i] = 1
            else:
                self.labels[i] = 0

    def get_data_and_labels(self):
        return self.data, self.labels

    def plot_data(self):
        plt.figure(figsize=(8, 6))
        # Plot points with different colors based on their labels
        plt.scatter(self.data[self.labels == 1][:, 0], self.data[self.labels == 1][:, 1], color='blue', label='Classe 1')
        plt.scatter(self.data[self.labels == 0][:, 0], self.data[self.labels == 0][:, 1], color='red', label='Classe 0')
        
        # Plot the line
        x_vals = np.array([0, 1])
        y_vals = self.slope * x_vals + self.intercept
        plt.plot(x_vals, y_vals, color='black', linestyle='--', label='Função Geradora')
        
        plt.xlabel('$x_1$')
        plt.ylabel('$x_2$')
        plt.title('Dados gerados e Função Geradora')
        plt.legend()
        plt.show()

class Perceptron:
    def __init__(self, input_size, activation_function, learning_rate=0.01):
        self.weights = np.zeros(input_size + 1)
        self.activation_function = activation_function
        self.learning_rate = learning_rate

    def activation(self, x):
        if self.activation_function == 'relu':
            return max(0, x)
        elif self.activation_function == 'sigmoid':
            return 1 / (1 + np.exp(-x))

    def predict(self, inputs):
        inputs_with_bias = np.append(inputs, 1)
        weighted_sum = np.dot(self.weights, inputs_with_bias)
        return self.activation(weighted_sum)

    def train(self, training_data, labels, epochs=100):
        for _ in range(epochs):
            for inputs, label in zip(training_data, labels):
                prediction = self.predict(inputs)
                error = label - prediction
                if self.activation_function == 'step':
                    update = self.learning_rate * error * inputs
                    self.weights[:-1] += update
                    self.weights[-1] += self.learning_rate * error
                else:
                    update = self.learning_rate * error * prediction * (1 - prediction) * inputs
                    self.weights[:-1] += update
                    self.weights[-1] += self.learning_rate * error * prediction * (1 - prediction)
        return self.weights

# Example usage
if __name__ == "__main__":
    data_gen = DataGenerator(num_points=200)
    training_data, labels = data_gen.get_data_and_labels()
    data_gen.plot_data()

    perceptron_relu = Perceptron(input_size=2, activation_function='relu')
    weights_relu = perceptron_relu.train(training_data, labels, epochs=10)
    print(f"Pesos finais (ReLU): {weights_relu}")

    perceptron_sigmoid = Perceptron(input_size=2, activation_function='sigmoid')
    weights_sigmoid = perceptron_sigmoid.train(training_data, labels, epochs=10)
    print(f"Pesos finais (Sigmoid): {weights_sigmoid}")
