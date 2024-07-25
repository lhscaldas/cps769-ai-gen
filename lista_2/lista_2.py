import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import random

class NeuralNetworkModel:
    def __init__(self, data_path, seed=42):
        self.data_path = data_path
        self.model = None
        self.history = None
        self.scaler = StandardScaler()
        self.seed = seed
        self._set_seed()
    
    def _set_seed(self):
        np.random.seed(self.seed)
        random.seed(self.seed)
        tf.random.set_seed(self.seed)
    
    def load_data(self):
        self.data = pd.read_csv(self.data_path, decimal=',', dtype=float)
        self.X = self.data[['Presença', 'HorasEstudo']]
        self.y = self.data['Nota']
    
    def preprocess_data(self):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=self.seed)
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)
    
    def build_model(self):
        self.model = tf.keras.Sequential([
            tf.keras.layers.InputLayer(input_shape=(2,)),
            tf.keras.layers.Dense(3, activation='relu'),
            tf.keras.layers.Dense(1)
        ])
        self.model.compile(optimizer='adam', loss='mean_squared_error')
    
    def train_model(self, epochs=600):
        self.history = self.model.fit(self.X_train_scaled, self.y_train, epochs=epochs, verbose=0)
    
    def evaluate_model(self):
        test_loss = self.model.evaluate(self.X_test_scaled, self.y_test)
        print(f'Mean Squared Error on Test Data: {test_loss}')
    
    def plot_loss(self):
        plt.plot(self.history.history['loss'], label='Train Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Mean Squared Error')
        plt.legend()
        plt.show()
    
    def show_weights_and_biases(self):
        for layer in self.model.layers:
            weights, biases = layer.get_weights()
            print(f'Pesos da camada: {weights}')
            print(f'Bias da camada: {biases}')
    
    def predict_new_data(self, new_data_path):
        new_data = pd.read_csv(new_data_path, decimal='.', dtype=float)
        new_X = new_data[['Presença', 'HorasEstudo']]
        new_X_scaled = self.scaler.transform(new_X)
        predictions = self.model.predict(new_X_scaled, verbose=0)
        new_data['Nota'] = predictions
        new_data.to_csv('lista_2/lista_2-students_data_new.csv', index=False)
        print(f'Previsões salvas no arquivo lista_2-students_data_new.csv')
    
    def modify_and_train_new_model(self, new_layers, epochs=600):
        modified_model = tf.keras.Sequential()
        modified_model.add(tf.keras.layers.InputLayer(input_shape=(2,)))
        
        for units in new_layers:
            modified_model.add(tf.keras.layers.Dense(units, activation='relu'))
        
        modified_model.add(tf.keras.layers.Dense(1))
        modified_model.compile(optimizer='adam', loss='mean_squared_error')
        
        modified_history = modified_model.fit(self.X_train_scaled, self.y_train, epochs=epochs, verbose=0)
        modified_test_loss = modified_model.evaluate(self.X_test_scaled, self.y_test)
        print(f'Mean Squared Error on Test Data with Modified Model: {modified_test_loss}')

if __name__ == "__main__":
    data_path = 'lista_2/lista_2-students_data.csv'
    new_data_path = 'lista_2/lista_2-students_data_new.csv'

    nn_model = NeuralNetworkModel(data_path)
    nn_model.load_data()
    nn_model.preprocess_data()
    nn_model.build_model()
    nn_model.train_model()
    nn_model.evaluate_model()
    # nn_model.plot_loss()
    # nn_model.show_weights_and_biases()
    # nn_model.predict_new_data(new_data_path)
    nn_model.modify_and_train_new_model([3, 3, 3])
    nn_model.modify_and_train_new_model([9])
    nn_model.modify_and_train_new_model([100])
