import numpy as np


class NeuralNet:
    def __init__(self, hidden_layer_size: int, categories: int, num_features: int):
        self.W1 = np.random.rand(hidden_layer_size, num_features) - 0.5
        self.b1 = np.random.rand(hidden_layer_size, 1) - 0.5
        self.W2 = np.random.rand(categories, hidden_layer_size) - 0.5
        self.b2 = np.random.rand(categories, 1) - 0.5

    @staticmethod
    def ReLU(Z: np.ndarray) -> np.ndarray:
        return np.maximum(Z, 0)

    @staticmethod
    def softmax(Z: np.array) -> np.array:
        return np.exp(Z) / sum(np.exp(Z))

    def forward_prop(self, X: np.ndarray) -> np.ndarray:
        self.Z1: np.ndarray = (
            self.W1.dot(X) + self.b1
        )  # W1 10,784 ||| X 784,60000 ||| W.X 10,60000
        self.A1: np.ndarray = NeuralNet.ReLU(self.Z1)
        self.Z2: np.ndarray = self.W2.dot(self.A1) + self.b2
        self.A2: np.ndarray = NeuralNet.softmax(self.Z2)
        return self.A2

    @staticmethod
    def one_hot(Y: np.ndarray) -> np.ndarray:
        one_hot_Y = np.zeros((Y.size, Y.max() + 1))  # create matrix with correct size
        # np.arange(Y.size) - will return a list from 0 to Y.size
        # Y - will contain the column index to set the value of 1
        one_hot_Y[np.arange(Y.size), Y] = 1
        return one_hot_Y.T  # transpose

    @staticmethod
    def deriv_ReLU(Z: np.ndarray) -> np.ndarray:
        return Z > 0

    def back_prop(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        one_hot_Y: np.ndarray,
        alpha: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        m = Y.size
        dZ2 = self.A2 - one_hot_Y
        dW2 = 1 / m * dZ2.dot(self.A1.T)
        db2 = 1 / m * np.sum(dZ2)
        dZ1: np.ndarray = self.W2.T.dot(dZ2) * NeuralNet.deriv_ReLU(self.Z1)
        dW1 = 1 / m * dZ1.dot(X.T)
        db1 = 1 / m * np.sum(dZ1)

        self.W1 = self.W1 - alpha * dW1
        self.b1 = self.b1 - alpha * db1
        self.W2 = self.W2 - alpha * dW2
        self.b2 = self.b2 - alpha * db2

    @staticmethod
    def get_predictions(A2: np.ndarray) -> np.ndarray:
        return np.argmax(A2, 0)

    @staticmethod
    def get_correct_prediction(predictions: np.ndarray, Y: np.ndarray) -> float:
        return np.sum(predictions == Y)

    @staticmethod
    def get_accuracy(correct_prediction: int, size: int):
        return 1.0 * correct_prediction / size
