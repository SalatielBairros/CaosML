# KNN (K Nearest Neighbor)

## Introdução

O algoritmo KNN é baseado no princípio de que pontos próximos em um espaço de características tendem a ter rótulos semelhantes. Ele pode ser usado tanto para classificação quanto para regressão. Na classificação, o KNN atribui a uma instância de teste a classe mais frequente entre seus k vizinhos mais próximos. Na regressão, ele estima um valor pela média (ou outra medida) dos valores de seus vizinhos mais próximos.

Uma das principais características desse algoritmo é a simplicidade do uso. Por ser um "lazy learning", o processo de treinamento prévio é quase inexistente, dependendo dos dados enviados. Por causa disso, porém, esse algoritmo possui algumas desvantagens, como:

* Custo computacional que cresce diretamente relacionado com a quantidade de dimensões e dados;
* Alguns formatos de dados mais convexos não apresentação boa performance.

## Implementação

O passo-a-passo da implementação do KNN é bastante simples.

1. Calcula a distância entre um ponto e todos os pontos do dataset. Essa etapa pode ser otimizada com algoritmos que evitam com que o cálculo seja com todos os pontos sempre. Normalmente as otimizações utilizam alguma estrutura de grafo ou árvore. A distância pode ser a euclidiana ou manhattam
2. Escolhe-se os `k` elementos com menor distância do ponto escolhido.
3. Realiza-se uma votação. Existem vários algoritmos possíveis de votação. A implementação mais simples (e a escolhida aqui) é retornar a label mais frequente dentre os vizinhos.

Dessa forma, a implementação fica:

```python
def predict(self, x: np.array):
    predictions = []

    for item in x:
        distances = self.__calculate_distances__(item)
        distances_with_labels = np.c_[distances, self.labels]
        nearest = self.__nearest_neighborns__(distances_with_labels)
        prediction = self.__voting__(nearest[:,1])
        predictions.append(prediction)
    
    return np.asarray(predictions)
```

## Utilizando o algoritmo

Segue abaixo um exemplo de uso do algoritmo:

```python
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Normalizer
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from classification.knn.knn_classifier import KNNClassifier

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

knn_model = KNNClassifier(k=5)
train_predictions = knn_model.fit_predict(x_train, y_train)
train_score = accuracy_score(y_train, train_predictions)
test_predictions = knn_model.predict(x_test)
test_score = accuracy_score(y_test, test_predictions)

print(f'[LOCAL EUCLIDEAN] Train score: {train_score}')
print(f'[LOCAL EUCLIDEAN] Test score: {test_score}')

knn_model = KNNClassifier(k=5, distance='manhattan')
train_predictions = knn_model.fit_predict(x_train, y_train)
train_score = accuracy_score(y_train, train_predictions)
test_predictions = knn_model.predict(x_test)
test_score = accuracy_score(y_test, test_predictions)

print(f'[LOCAL MANHATTAN] Train score: {train_score}')
print(f'[LOCAL MANHATTAN] Test score: {test_score}')

```

O output da execução acima é:

```bash
[SKLEARN] Train score: 0.9666666666666667
[SKLEARN] Test score: 1.0
[LOCAL EUCLIDEAN] Train score: 0.9666666666666667
[LOCAL EUCLIDEAN] Test score: 1.0
[LOCAL MANHATTAN] Train score: 0.9666666666666667
[LOCAL MANHATTAN] Test score: 1.0
```

Algumas observações sobre a execução:

* É extremamente importante que os dados estejam normalizados. A diferença de escala afeta diretamente as medidas de distância (especialmente a euclidiana).
* Note como a implementação do Scikit Learn possui a mesma resposta para o dataset iris. A diferença ficará sempre no tempo de execução, visto que a lib `sklearn` é melhor otimizada, especialmente no cálculo das distâncias.
* Sobre as medidas de distância e quando escolher cada uma, assista a [esta aula](https://www.youtube.com/watch?v=h0e2HAPTGF4&t=2362s&ab_channel=MITOpenCourseWare) de introdução ao aprendizado de máquina do MIT.
* Para realizar a chamada utilizando regressão, basta configurar o parâmetro em `KNNClassifier(k=5, task='regression')`.
