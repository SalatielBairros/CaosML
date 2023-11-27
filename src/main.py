import numpy as np
import matplotlib.pyplot as plt

X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1)

np.random.seed(42)

n = 0.1 #taxa de aprendizado
iterations = 1000
m = 100 #nº de instâncias 
# 100,2
X_b = np.c_[np.ones((100, 1)), X]
X_new = np.array([[0], [2]])
X_new_b = np.c_[np.ones((2, 1)), X_new]
theta = np.random.randn(2, 1) #inicialização aleatória

for i in range(iterations):
    gradients = 2/m * X_b.T.dot(X_b.dot(theta) - y)
    theta = theta - n * gradients