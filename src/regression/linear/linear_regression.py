import numpy as np
import matplotlib.pyplot as plt


class LinearRegression:

    def __init__(self, random_state: int = None):
        self.coeficients = None
        self.intercept = None
        self.theta_param = None
        self.random_state = random_state
        self.gradient_steps = []

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
        y = 4 + 3 * x[:, 0].reshape(100, 1) + np.random.randn(size, 1)

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

    def fit_gradient_descent(self,
                             x: np.array,
                             y: np.array,
                             learning_rate: float = 0.1,
                             nro_iteractions: int = 100,
                             plot_result: bool = False) -> np.array:
        theta = np.random.randn(x.shape[1] + 1, 1)
        size = x.shape[0]
        x_bias = np.c_[np.ones((size, 1)), x]
        self.gradient_steps = [theta]

        for _ in range(nro_iteractions):
            gradients = 2/size * x_bias.T.dot(x_bias.dot(theta) - y)
            theta = theta - learning_rate * gradients
            self.gradient_steps.append(theta)

        self.theta_param = theta
        self.intercept = theta[0]
        self.coeficients = theta[1:]

        if plot_result:
            self.__plot_result_line__(x, y)

        return self.theta_param
    
    def fit_stochastic_gradient_descent(self,
                                        x: np.array,
                                        y: np.array,
                                        initial_learning_rate: float = 0.1,
                                        nro_iteractions: int = 100,
                                        plot_result: bool = False,
                                        decay: float = 0.01) -> np.array:
        theta = np.random.randn(x.shape[1] + 1, 1)
        size = x.shape[0]
        x_bias = np.c_[np.ones((size, 1)), x]
        self.gradient_steps = [theta]

        for epoch in range(nro_iteractions):
            for i in range(size):
                random_index = np.random.randint(size)
                xi = x_bias[random_index:random_index+1]
                yi = y[random_index:random_index+1]

                # gradients = 2/size * xi.T.dot(xi.dot(theta) - yi) when size == 1, so:
                gradients = 2 * xi.T.dot(xi.dot(theta) - yi)
                learning_rate = self.__learning_schedule__(initial_learning_rate, decay, epoch * i)
                theta = theta - learning_rate * gradients
                self.gradient_steps.append(theta)

        self.theta_param = theta
        self.intercept = theta[0]
        self.coeficients = theta[1:]

        if plot_result:
            self.__plot_result_line__(x, y)

        return self.theta_param

    def get_edges(self, x: np.array) -> np.array:
        n_features = x.shape[1]
        return np.array(
            [np.full((1, n_features), x.min())[0], np.full((1, n_features), x.max())[0]])
    
    def predict(self, x: np.array) -> np.array:
        x_bias = np.c_[np.ones((x.shape[0], 1)), x]
        return x_bias.dot(self.theta_param)
    
    def plot_all_theta_steps(self, x: np.array, y: np.array):
        to_predict = self.get_edges(x)        

        plt.scatter(x[:, 0], y, color='blue', label='Original data')
        for _, theta in enumerate(self.gradient_steps):
            plt.plot(to_predict[:, 0], theta[0] + theta[1] * to_predict[:, 0], "r-", alpha=0.1)
        plt.xlabel('Feature 1')
        plt.ylabel('Target Variable')
        plt.legend()
        plt.show()
    
    def __learning_schedule__(self, initial_learning_rate: float, decay: float, iteraction: int) -> float:
        return initial_learning_rate / (1 + decay * iteraction)

    def __plot_result_line__(self, x: np.array, y: np.array):
        to_predict = self.get_edges(x)
        y_predict = self.predict(to_predict)

        plt.scatter(x[:, 0], y, color='blue', label='Original data')
        plt.plot(to_predict[:, 0], y_predict, "r-", label='Predicted data')
        plt.xlabel('Feature 1')
        plt.ylabel('Target Variable')
        plt.legend()
        plt.show()


lr = LinearRegression(random_state=42)
a, b = lr.generate_random_linear_data(100, 1)

lr.fit_stochastic_gradient_descent(a, b, initial_learning_rate=0.1, nro_iteractions=30, plot_result=False)

lr.plot_all_theta_steps(a, b)
