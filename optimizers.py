import numpy as np

class Optimizer:
    """Base class for optimization algorithms."""
    def __init__(self, model, loss_fn, learning_rate=0.01, lambda_l1=0.0, lambda_l2=0.0, momentum = 0.9):
        self.model = model
        self.loss_fn = loss_fn
        self.learning_rate = learning_rate
        self.lambda_l1 = lambda_l1
        self.lambda_l2 = lambda_l2
        self.momentum = momentum
        self.velocity = {param_name: np.zeros_like(param_value) for param_name, param_value in model.get_parameters().items()}
    
    def compute_gradient(self, X, Y, epsilon=1e-5):
        """One-sided finite-difference gradient approximation (forward difference)."""
        gradients = {}
        initial_loss = self.loss_fn(Y, self.model.forward(X))
        
        for param_name, param_value in self.model.get_parameters().items():
            gradient = np.zeros_like(param_value)
            for i in range(param_value.shape[0]):
                for j in range(param_value.shape[1]):
                    original_value = param_value[i, j]
                    param_value[i, j] = original_value + epsilon
                    loss_plus = self.loss_fn(Y, self.model.forward(X))
                    param_value[i, j] = original_value  # Restore original value
                    gradient[i, j] = (loss_plus - initial_loss) / epsilon  # Forward difference
            
            gradients[param_name] = gradient
        return gradients

    def l1_regularization(self):
        """Computes L1 regularization loss."""
        return self.lambda_l1 * np.sum([np.sum(np.abs(param)) for param in self.model.get_parameters().values()])
    
    def l2_regularization(self):
        """Computes L2 regularization loss."""
        return self.lambda_l2 * np.sum([np.sum(param ** 2) for param in self.model.get_parameters().values()])    

    def step(self, X, Y, batch_size=32):
        """Performs a single optimization step using mini-batches with Momentum."""
        # Select a random batch
        indices = np.random.choice(X.shape[0], batch_size, replace=False)
        X_batch = X[indices]
        Y_batch = Y[indices]
        
        # Compute gradients on the batch
        gradients = self.compute_gradient(X_batch, Y_batch)
        
        # Update parameters using momentum
        for param_name, grad in gradients.items():
            reg_term = 0
            if self.lambda_l1 > 0:
                reg_term += self.l1_regularization()
            if self.lambda_l2 > 0:
                reg_term += self.l2_regularization()

            # Momentum update
            self.velocity[param_name] = self.momentum * self.velocity[param_name] + (1 - self.momentum) * grad

            # Parameter update
            self.model.params[param_name] -= self.learning_rate * (self.velocity[param_name] + reg_term)

    def early_stopping(self, val_loss, patience=5):
        """Implements early stopping based on validation loss."""
        if not hasattr(self, 'best_loss'):
            self.best_loss = np.inf
            self.counter = 0
        if val_loss < self.best_loss:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
        return self.counter >= patience
    
#Loss Functions used as static functions

def mean_squared_error(y_true, y_pred): 
        return np.mean((y_true - y_pred) ** 2)

def binary_cross_entropy(y_true, y_pred):
    epsilon = 1e-12  # To prevent log(0)
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

def categorical_cross_entropy(y_true, y_pred):
    epsilon = 1e-12
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    return -np.sum(y_true * np.log(y_pred)) / y_true.shape[0]