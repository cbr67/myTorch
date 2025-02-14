import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, mean_absolute_error, mean_squared_error, root_mean_squared_error, r2_score, max_error

class Report:
    """Class for generating performance reports and visualizations."""
    def __init__(self, model, X_test, y_test, y_pred, task):
        self.model = model
        self.X_test = X_test
        self.y_test = y_test
        self.y_pred = y_pred
        self.task = task
    
    def loss_curve(self, train_losses, val_losses):
        """Plot training and validation loss curves."""
        plt.plot(train_losses, label='Training Loss')
        plt.plot(val_losses, label='Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.title('Loss Curve')
        plt.show()

    def parity_plot(self):
        """Plot a parity plot for regression tasks."""
        plt.scatter(self.y_test, self.y_pred)
        plt.xlabel('True Values')
        plt.ylabel('Predicted Values')
        plt.title('Parity Plot')
        plt.plot([self.y_test.min(), self.y_test.max()], [self.y_test.min(), self.y_test.max()], 'r')
        plt.show()
  
    def confusion_matrix_plot(self):
        """Plot confusion matrix for classification tasks."""
        cm = confusion_matrix(self.y_test, self.y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title('Confusion Matrix')
        plt.show()
    
    def classification_metrics(self):
        """Compute and return classification metrics as a dictionary."""
        return classification_report(self.y_test, self.y_pred, output_dict=True)

    def regression_metrics(self):
        """Returns common regression metrics in a dictionary."""
        return {
            "MAE": round(mean_absolute_error(self.y_test, self.y_pred),2),
            "MSE": round(mean_squared_error(self.y_test, self.y_pred),2),
            "RMSE": round(root_mean_squared_error(self.y_test, self.y_pred),2),
            "R-squared": round(r2_score(self.y_test, self.y_pred),2),
            "MAPE": round(np.mean(np.abs((self.y_test - self.y_pred) / self.y_test)) * 100, 2),
            "Max Error": round(max_error(self.y_test, self.y_pred), 2)
        }

    def regression_plot_2d(self):
        # Plot regression line
        plt.scatter(self.X_test, self.y_test, label="Actual Data")
        plt.plot(self.X_test, self.y_pred, color='red', label="Regression Line")
        plt.xlabel("Feature")
        plt.ylabel("Target")
        plt.title("Linear Regression Performance")
        plt.legend()
        plt.show()
    

    def regression_plot_3d(self):
        """
        Plots a 3D regression model with two input features and a regression plane.
        """
        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111, projection='3d')

        # Extract features
        x1 = self.X_test[:, 0]  # First feature
        x2 = self.X_test[:, 1]  # Second feature

        # Scatter actual data points
        ax.scatter(x1, x2, self.y_test, color='blue', label='Actual Data')

        # Create a grid over the feature space
        x1_range = np.linspace(x1.min(), x1.max(), 20)
        x2_range = np.linspace(x2.min(), x2.max(), 20)
        x1_grid, x2_grid = np.meshgrid(x1_range, x2_range)
        
        # Predict values for the grid to form a regression plane
        X_grid = np.column_stack((x1_grid.ravel(), x2_grid.ravel()))
        y_grid_pred = self.model.forward(X_grid).reshape(x1_grid.shape)

        # Plot the regression plane
        ax.plot_surface(x1_grid, x2_grid, y_grid_pred, color='red', alpha=0.5)

        # Labels and title
        ax.set_xlabel('Feature 1')
        ax.set_ylabel('Feature 2')
        ax.set_zlabel('Target')
        ax.set_title('3D Regression Plot with Plane')

        # Legend
        ax.legend()
        
        # Show plot
        plt.show()
