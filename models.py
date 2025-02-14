import numpy as np

class Model:
    """Base class for all models."""
    def __init__(self, input_dim, output_dim):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.params = {
            #'W': np.random.randn(output_dim, input_dim),
            'W': np.random.randn(input_dim, output_dim),
            'b': np.random.randn(output_dim, 1)
        }
    
    def forward(self, x):
        #return np.dot(self.params['W'], x) + self.params['b']
        return np.dot(x, self.params['W']) + self.params['b']
    
    def get_parameters(self):
        return self.params

class LinearModel(Model):
    pass #base model is linear

class LogisticRegression(Model):
    def forward(self, x):
        return 1 / (1 + np.exp(-super().forward(x)))


class DenseNetwork(Model):
    """Implements a feed-forward neural network compatible with the Optimizer."""
    def __init__(self, layers, activation='relu', output_activation=None):
        super().__init__(layers[0], layers[-1])
        self.layers = layers
        self.activation = activation
        self.output_activation = output_activation
        
        # Store weights and biases in self.params for Optimizer compatibility
        self.params = {}
        for i in range(len(layers)-1):
            self.params[f"W{i}"] = np.random.randn(layers[i], layers[i+1]) * np.sqrt(2 / layers[i])
            self.params[f"b{i}"] = np.zeros((1, layers[i+1]))

    def get_parameters(self):
        """Return parameters in a dictionary format."""
        return self.params

    def relu(self, z):
        return np.maximum(0, z)
    
    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))
    
    def softmax(self, z):
        exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)

    def forward(self, x):
        """Performs a forward pass through the network."""
        a = x

        for i in range(len(self.layers) - 1):
            W = self.params[f"W{i}"]
            b = self.params[f"b{i}"]
            z = np.dot(a, W) + b
            
            if i < len(self.layers) - 2:
                a = self.relu(z) if self.activation == 'relu' else self.sigmoid(z)
            else:
                a = self.softmax(z) if self.output_activation == 'softmax' else z

        return a



