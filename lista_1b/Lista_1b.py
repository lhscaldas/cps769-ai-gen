import numpy as np

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
training_data = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])
labels = np.array([0, 0, 0, 1])

perceptron_relu = Perceptron(input_size=2, activation_function='relu')
weights_relu = perceptron_relu.train(training_data, labels, epochs=10)
print(f"Pesos finais (ReLU): {weights_relu}")

perceptron_sigmoid = Perceptron(input_size=2, activation_function='sigmoid')
weights_sigmoid = perceptron_sigmoid.train(training_data, labels, epochs=10)
print(f"Pesos finais (Sigmoid): {weights_sigmoid}")
