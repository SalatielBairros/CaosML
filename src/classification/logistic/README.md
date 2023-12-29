# Regressão Logística

## Introdução

A regressão logística é um modelo de classificação que pode ser usado para prever a probabilidade de ocorrência de um evento. A regressão logística é um modelo de classificação binária, ou seja, a variável resposta assume apenas dois valores, 0 ou 1. A regressão logística é um caso especial da regressão linear, onde a variável resposta é binária. A regressão logística é usada para modelar a probabilidade de uma resposta ocorrer em função de um conjunto de variáveis explicativas (preditoras).

Existem algums pré requisitos para utilizar a regressão logística, são eles:

- A variável resposta deve ser binária (0 ou 1);
- As observações devem ser independentes entre si (ou seja, pouca multicolinearidade);
- As observações devem ser representativas (ou seja, não podem ser outliers);
- As observações devem ser grandes o suficiente para garantir um número mínimo de ocorrências de cada classe.

> É necessário lembrar que as implementações daqui não têm como objetivo a melhor performance ou otimização computacional. O objetivo das implementações e explicações é didático, por isso é feito da forma mais simples possível, sem incluir, por exemplo, otimizações no _learning rate_ ou penalizações (_l1_, _l2_ e outras). Para entender mais as opções da regressão logística, veja a implementação da biblioteca Scikit-Lear [aqui](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html).

## Explicação matemática

Existe uma similaridade bastante grande entre a Regressão Logística (RLog) e a Regressão Linear (RLin). Com isso, serão consideradas explicações já realizadas para o algoritmo de Regressão Linear.

A diferença entre a RLog e a RLin está no objetivo. A ideia é ter uma resposta binária, 0 ou 1. Com isso, ainda que o nome seja _regressão_, o objetivo é uma _classificação_, ou seja, é um algoritmo supervisionado de classificação. Existem dois principais métodos de fazer isso:

1. Minimizar a função de custo (_loss_) -> Gradiente descendente
2. Maximizar a função de _likehood_ -> Gradiente ascendente

### Minimizando _loss_

O processo de minimizar a função _loss_ é bastante semelhante ao que acontece na RLin. Sendo $\theta$ o vetor dos coeficientes referentes aos atributos dos dados mais o bias, ficamos com a função linear conhecida $Z = X \cdot \theta$.

Contudo, a saída do nosso modelo não é uma predição de um próximo valor, mas uma classificação entre 0 e 1 do mesmo. Para tal, é utilizada a função _sigmóide_, que coloca os valores resultantes entre 0 e 1. A função sigmóide é representada por:

$$
\sigma(Z) = \frac{1}{1 + e^{-z}}
$$

Com isso, o objetivo é ajustar os parâmetros $\theta$ de forma que o resultado da sigmóide fique o mais próximo do esperado, ou seja, mais próximo de $1$ quando $y_n = 1$ e mais próximo de %0% quando $y_n = 0$.

Uma das formas de se chegar nisso é minimizar a função de custo que representa as diferenças entre o esperado e o resultado. O processo de se chegar na função de custo já foi feito para a RLin. Abaixo segue a função de custo em termos de $\theta$

$$
loss(\theta) = \frac{-Y^T\cdot log(h) - (1 - Y)^T\cdot log(1 - h)}{m}
$$

Note que de um lado temos $-Y^T$ e do outro $(1 - Y)^T$. O primeiro caso é eliminado quando $y = 0$ e o segundo quando $y = 1$. É o ajuste para que fique uma equação única em função de $\theta$.

O gradiente é, como já vimos, a derivada parcial em função de $\theta$ para a função de _loss_. Dessa forma podemos tentar achar o ponto mínimo da função de erro. A derivada é:

$$
\frac{\delta loss(\theta)}{\delta(\theta)} = \frac{X^T\cdot (H - Y)}{m}
$$

Com isso, o funcionamento do Gradiente Descendente é o mesmo já conhecido. Atualizamos o $\theta$ subtraindo o resultado do gradiente vezes a taxa de aprendizado (_learning rate_). Taxa essa que pode ser otimizada de diversas formas, mas que as otimizações não estão implementadas aqui neste projeto.

$$
\theta = \theta - \alpha \cdot \frac{\delta loss(\theta)}{\delta(\theta)}
$$

Após cada atualização, o processo é feito novamente, buscando novos valores da função _sigmóide_, até que o número máximo de tentativas (hiperparâmetro) seja atingido.

#### Implementação Python

A implementação em python é conforme abaixo:

```python
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
```

> Para ver com detalhes as implementações das funções chamadas, veja o código fonte do projeto.

### Minimizando _likehood_

A Máxima Verossimilhança (Maximum Likelihood Estimation - MLE) para a Regressão Logística é um método utilizado para encontrar os valores dos parâmetros do modelo que maximizam a função de verossimilhança. A ideia fundamental por trás da MLE é escolher os parâmetros que tornam mais provável observar os dados que temos.

O funcionamento matemático é quase o mesmo. A diferença é que o objetivo é encontrar o ponto _máximo_ da função. Com isso, o gradiente é _somado_:

$$
\theta = \theta + \alpha \cdot \frac{\delta loss(\theta)}{\delta(\theta)}
$$

A função MLE (em sua variação log MLE) é expressa na equação abaixo. Não é o objetivo aqui explicar detalhadamente como chegamos nessa função, mas é bastante conhecida dentro da área da estatística:

$$
ll = y \cdot X\cdot\theta - log(1 + e^{X\cdot\theta})
$$

A derivada correspondente é:

$$
\Delta ll = X^T(y - Z)
$$

A implementação em Python, conforme pode ser visto no código fonte do projeto, é igual, apenas alterando a função de gradiente e a função de atualização dos pesos $\theta$.

## Utilizando o algoritmo

Abaixo segue um código com exemplo de uso do algoritmo:

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from classification.logistic.logistic_regression import LogisticRegression
from sklearn.linear_model import LogisticRegression as SKLogisticRegression
import time

# Carrega um dataset de exemplo direto do GitHub. O dataset simula dados de alunos em duas provas e o resultado se foram aprovados ou reprovados no final.
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
```

O código abaixo terá o output:

```bash
Training with loss_minimization:
Result: 0.89 in 68.81536173820496 seconds

Training with maximum_likehood
Result: 0.87 in 59.98377537727356 seconds

Training with SKLearn
Result: 0.87 in 0.0 seconds
```

Note como a diferença entre a implementação local e o SkLearn é o tempo de execução e não a performance. Isso acontece porque muitas otimizações são feitas e até mesmo o método utilizado pelo SkLearn não é exatamente o gradiente.

## Conclusão¹

Neste documento, exploramos os detalhes matemáticos da regressão logística, concentrando-nos nas abordagens do gradiente descendente e ascendente. O gradiente descendente destaca-se pela sua eficiência em ajustar os parâmetros do modelo, minimizando a função de perda. Por outro lado, a maximização da verossimilhança oferece uma perspectiva centrada na probabilidade dos dados observados.
 
Se quiser saber mais, explore melhorias como:

1. Regressão Logística Regularizada:

Estude variantes da regressão logística que incorporam termos de regularização, como Ridge (L2) e Lasso (L1). Isso pode fortalecer a resistência do modelo a overfitting.

2. Regressão Logística Multinomial:

Explore a regressão logística multinomial para problemas de classificação com mais de duas classes. Isso amplia a aplicabilidade do modelo.

Espero que este projeto tenha sido alguma ajuda no entendimento da Regressão Logística.

----

<sup><sup>1. Conclusão (e apenas ela) escrita com o auxílio do Chat GPT 3</sup></sup>