import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from classification.logistic.logistic_regression import LogisticRegression
from sklearn.linear_model import LogisticRegression as SKLogisticRegression
import time

dataset = pd.read_csv('https://raw.githubusercontent.com/animesh-agarwal/Machine-Learning/master/LogisticRegression/data/marks.txt', 
                      header=None, names=['Exam 1', 'Exam 2', 'Admitted'])

X = dataset[['Exam 1', 'Exam 2']].copy()
y = dataset['Admitted'].copy()

labels, index = np.unique(dataset["Admitted"], return_inverse=True)
plt.scatter(dataset['Exam 1'], dataset['Exam 2'], marker='o', c=index)
plt.show()

lr = LogisticRegression(random_state=42)
print("Training with loss_minimization:")
start_time = time.time()
result = (lr
      .fit(X, y, learning_rate=0.001, epochs=1000000, calculate_cost=False, save_steps=False, random_inicialization=True, method='loss_minimization')
      .accuracy(X, y))
print(f'Result: {result} in {str(time.time() - start_time)} seconds')

print("\n Training with maximum_likehood")
start_time = time.time()
result = (lr
      .fit(X, y, learning_rate=0.001, epochs=1000000, calculate_cost=False, save_steps=False, random_inicialization=True, method='maximum_likehood')
      .accuracy(X, y))
print(f'Result: {result} in {str(time.time() - start_time)} seconds')

print("\n Training with SKLearn")
start_time = time.time()
sk_lr = SKLogisticRegression(penalty=None, random_state=42, max_iter=1000000)
sk_lr.fit(X, y)
score = sk_lr.score(X, y)
print(f'Result: {result} in {str(time.time() - start_time)} seconds')
