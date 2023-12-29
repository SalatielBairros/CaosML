import numpy as np

class LogisticRegression:
    def __init__(self, random_state: int = None):
        self.theta = None
        self.theta_steps = []
        self.random_state = random_state

        if self.random_state is not None:
            np.random.seed(self.random_state)

    def fit(self, 
            x: np.array, 
            y: np.array, 
            epochs: int = 1000, 
            calculate_cost: bool = True, 
            learning_rate: float = 0.01, 
            method: str = 'loss_minimization', 
            save_steps: bool = True,
            random_inicialization: bool = False):
        if method == 'loss_minimization':
            self.__fit_loss_minimization__(x, y, epochs, calculate_cost, learning_rate, save_steps, random_inicialization)
        elif method == 'maximum_likehood':
            self.__fit_max_likehood__(x, y, epochs, calculate_cost, learning_rate, save_steps, random_inicialization)
        else:
            raise NotImplementedError(f'Method {method} not implemented')
            
        return self
    
    def predict(self, x):
        x_bias = self.__add_bias_term__(x)
        hx = self.__hx__(x_bias)        
        predictions = np.vectorize(lambda x : 0 if x < 0.5 else 1)(hx)
        return predictions
    
    def accuracy(self, x: np.array, y: np.array):
        predictions = self.predict(x)
        return np.mean(predictions == y)
    
    def __fit_loss_minimization__(self, 
                                  x: np.array, 
                                  y: np.array, 
                                  epochs: int, 
                                  calculate_cost: bool, 
                                  learning_rate: float, 
                                  save_steps: bool,
                                  random_inicialization: bool):
        
        x_bias = self.__add_bias_term__(x)        
        self.theta = np.random.randn(x_bias.shape[1]) if random_inicialization else np.zeros(x_bias.shape[1])
        
        for i in range(epochs):
            hx = self.__hx__(x_bias)
            if calculate_cost and i % 100 == 0:
                cost = self.binary_cross_entropy_loss(y, hx)
                print(f'Cost for epoch {i} is {cost}')            
            gradient = self.__gradient_descent__(x_bias, hx, y)
            if save_steps:
                self.__save_current_theta__()
            self.__update_weight_loss__(learning_rate, gradient)

    def __fit_max_likehood__(self, 
                                  x: np.array, 
                                  y: np.array, 
                                  epochs: int, 
                                  calculate_cost: bool, 
                                  learning_rate: float, 
                                  save_steps: bool,
                                  random_inicialization: bool):
        
        x_bias = self.__add_bias_term__(x)        
        self.theta = np.random.randn(x_bias.shape[1]) if random_inicialization else np.zeros(x_bias.shape[1])
        
        for i in range(epochs):
            hx = self.__hx__(x_bias)
            if calculate_cost and i % 100 == 0:
                cost = self.binary_cross_entropy_loss(y, hx)
                print(f'Cost for epoch {i} is {cost}')            
            gradient = self.__gradient_ascent__(x_bias, hx, y)
            if save_steps:
                self.__save_current_theta__()
            self.__update_weight_mle__(learning_rate, gradient)

    def __save_current_theta__(self):
        self.theta_steps.append(self.theta)

    def __add_bias_term__(self, x: np.array) -> np.array:
        intercept = np.ones((x.shape[0], 1)) 
        return np.concatenate((intercept, x), axis=1)
    
    def __hx__(self, x: np.array) -> np.array:
        z = np.dot(x, self.theta)
        return 1 / (1 + np.exp(-z))    
    
    def binary_cross_entropy_loss(self, y_true: np.array, y_pred: np.array) -> float:
        epsilon = 1e-15  # Small constant to avoid log(0)
        new_y_pred = np.clip(y_pred, epsilon, 1 - epsilon)  # Clip to avoid log(0) and log(1)

        loss = -((y_true.dot(np.log(new_y_pred)) + (1-y_true).dot(np.log(1-new_y_pred))) / y_true.shape[0])
        return loss
    
    def __gradient_descent__(self, x: np.array, h: np.array, y: np.array):
        return x.T.dot((h - y)) / y.shape[0]        
    
    def __update_weight_loss__(self, learning_rate, gradient):
        self.theta = self.theta - (learning_rate * gradient)
    
    def __log_likelihood__(self, x, y, weights):
        z = np.dot(x, weights)
        ll = np.sum( y*z - np.log(1 + np.exp(z)) )
        return ll
    
    def __gradient_ascent__(self, x:np.array, h: np.array, y: np.array):
        return np.dot(x.T, y - h)
    
    def __update_weight_mle__(self, learning_rate, gradient):
        self.theta = self.theta + (learning_rate * gradient)
    