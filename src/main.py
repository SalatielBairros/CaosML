import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Normalizer
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from neighbors.knn.knn import KNN

SEED = 42
np.random.seed(SEED)

iris = datasets.load_iris()

x = iris['data']
y = iris['target']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, shuffle=True, random_state=SEED)

x_train = np.asarray(x_train)
x_test = np.asarray(x_test)
y_train = np.asarray(y_train)
y_test  = np.asarray(y_test)

scaler = Normalizer().fit(x_train)
normalized_x_train = scaler.transform(x_train)
normalized_x_test = scaler.transform(x_test)

sk_model = KNeighborsClassifier(n_neighbors=5, algorithm='brute')
sk_model.fit(x_train, y_train)

sk_train_predictions = sk_model.predict(x_train)
sk_test_predictions = sk_model.predict(x_test)

sk_train_score = accuracy_score(y_train, sk_train_predictions)
sk_test_score = accuracy_score(y_test, sk_test_predictions)

print(f'[SKLEARN] Train score: {sk_train_score}')
print(f'[SKLEARN] Test score: {sk_test_score}')

knn_model = KNN(k=5)
train_predictions = knn_model.fit_predict(x_train, y_train)
train_score = accuracy_score(y_train, train_predictions)
test_predictions = knn_model.predict(x_test)
test_score = accuracy_score(y_test, test_predictions)

print(f'[LOCAL EUCLIDEAN] Train score: {train_score}')
print(f'[LOCAL EUCLIDEAN] Test score: {test_score}')

knn_model = KNN(k=5, distance='manhattan')
train_predictions = knn_model.fit_predict(x_train, y_train)
train_score = accuracy_score(y_train, train_predictions)
test_predictions = knn_model.predict(x_test)
test_score = accuracy_score(y_test, test_predictions)

print(f'[LOCAL MANHATTAN] Train score: {train_score}')
print(f'[LOCAL MANHATTAN] Test score: {test_score}')
