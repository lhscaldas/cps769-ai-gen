import numpy as np

class Perceptron:
    def __init__(self, input_size, learning_rate=0.01):
        self.weights = np.zeros(input_size + 1)  # +1 for the bias weight
        self.learning_rate = learning_rate

    def activation_function(self, x):
        # Step activation function
        return 1 if x >= 0 else 0

    def predict(self, inputs):
        # Add bias term to inputs
        inputs_with_bias = np.append(inputs, 1)
        # Calculate the weighted sum
        weighted_sum = np.dot(self.weights, inputs_with_bias)
        # Apply activation function
        return self.activation_function(weighted_sum)

    def train(self, training_data, labels, epochs=100):
        for _ in range(epochs):
            for inputs, label in zip(training_data, labels):
                prediction = self.predict(inputs)
                error = label - prediction
                # Update the weights
                self.weights[:-1] += self.learning_rate * error * inputs
                self.weights[-1] += self.learning_rate * error  # Update bias weight

# Example usage
if __name__ == "__main__":
    # Sample training data (AND logic gate)
    training_data = np.array([
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1]
    ])
    labels = np.array([0, 0, 0, 1])

    perceptron = Perceptron(input_size=2)
    perceptron.train(training_data, labels, epochs=10)

    # Test the perceptron
    test_data = np.array([1, 1])
    print(f"Prediction for {test_data}: {perceptron.predict(test_data)}")
