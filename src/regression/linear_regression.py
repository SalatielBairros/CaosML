import numpy as np
import matplotlib.pyplot as plt

class LinearRegression:

    def __init__(self, random_state: int = None):
        self.coeficients = None
        self.intercept = None
        self.theta_param = None
        self.random_state = random_state

        if self.random_state is not None:
            np.random.seed(self.random_state)
    
    def generate_random_linear_data(self, size: int, num_features: int):
        """
        Generates random linear data.

        Args:
            size (int): The number of data points to generate.
            num_features (int): The number of features in the data.

        Returns:
            tuple: A tuple containing the generated input data (x) and the corresponding output data (y).
        """
        x = 2 * np.random.rand(size, num_features)
        y = 4 + 3 * x[:, 0]
        
        if x.shape[1] > 1:        
            y = y + 2 * x[:, 1]
            if x.shape[1] > 2:
                y = y + 1.5 * x[:, 2]
        y = y + np.random.randn(size)

        return x, y
    
    def fit_lsm(self, x: np.array, y: np.array, plot_result: bool = False) -> np.array:
        """
        Fits the linear regression model using the Least Squares Method (LSM).
        Parameters:
        - x: Input features as a numpy array.
        - y: Target values as a numpy array.
        - plot_result: Boolean flag indicating whether to plot the result or not.
        Returns:
        - theta_param: The learned parameters of the linear regression model.
        """
        size = x.shape[0]
        x_bias = np.c_[np.ones((size, 1)), x]
        theta_param = np.linalg.inv(x_bias.T.dot(x_bias)).dot(x_bias.T).dot(y)       
        
        self.theta_param = theta_param
        self.intercept = theta_param[0]
        self.coeficients = theta_param[1:]
        
        if plot_result:
            self.__plot_result_line__(x, y)
        
        return self.theta_param
    
    def __plot_result_line__(self, x: np.array, y: np.array):
        n_features = x.shape[1]            
        to_predict = np.array(
            [np.full((1, n_features), x.min())[0], np.full((1, n_features), x.max())[0]])
        x_new_bias = np.c_[np.ones((2, 1)), to_predict]
        y_predict = x_new_bias.dot(self.theta_param)

        plt.scatter(x[:, 0], y, color='blue', label='Original data')
        plt.plot(to_predict[:, 0], y_predict, "r-", label='Predicted data')
        plt.xlabel('Feature 1')
        plt.ylabel('Target Variable')
        plt.legend()
        plt.show()
