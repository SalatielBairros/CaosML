# Regressão Linear

Realiza uma aproximação linear dos dados observados em um dataset através da minimização da soma dos quadrados da diferença entre os valores do dataset (alvo) e preditos pela aproximação.

Para a regressão linear os dados precisam ser numéricos contínuos.

Existem duas principais formas de realizar a regressão linear: 

1. Utilizando LSM (least square method)
2. Gradiente Descendente

## LSM

Utiliza cálculo e álgebra linear para encontrar ponto mínimo de uma função de erro baseado na soma dos quadrados da diferença entre predito e real. Nesse ponto mínimo os coeficientes são encontrados para gerar a reta preditiva.

### Explicação matemática

Vamos começar lembrando a equação de uma função linear comum:

$f(x) = ax + b$

Onde: 

* $a$ é o coefiente angular da reta
* $b$ é a interseção da reta no eixo y. Também pode ser chamado de _bias_. 

Nos termos utilizados em Álgebra Linear para ML, a equação pode ser apresentada como:

$y = \theta_0 + \theta_1x$

No entanto, a equação acima se refere apenas a uma dimensão para atributos. Para $n$ dimensões, a equação é:

$y = \theta_0 + \theta_1x_1 + \theta_2x_x + ... + \theta_nx_n + \epsilon$

>Note a variável acrescentada $\epsilon$. Ela representa o erro, ou seja, a variabilidade da variável depentente ($y$) que não é explicada linearmente. Na regressão linear, é assumido que $\epsilon$ é distribuído normalmente com uma média zero e variância constante.

Diante disso, podemos considerar tanto os atributos ($x$) quanto os parâmetros ($\theta$) como vetores:

$X = [x_1, x_2, ..., x_n]$

$\theta = [\theta_0, \theta_1, \theta_2, ..., \theta_n]$

Com isso, a equação vetorial para a reta é:

$y = X . \theta + \epsilon$

O objetivo da regressão linear, portanto, é encontrar o melhor vetor $\theta$ que forme a reta com menor erro quadrático para os pontos do dataset.

O erro quadrático médio (MSE) é calculado somando o quadrado das diferenças entre o valor predito e o valor real e dividindo pelo total de pontos.

$$
MSE = \frac{\sum_{i=1}^{n}(y_{real} - y_{predicted})^2}{n}
$$ 

Onde:

$$
y_{predicted} = \theta_0 + \theta_1x
$$

Portanto:

$$
MSE = \frac{\sum_{i=1}^{n}(y_{real} - (\theta_0 + \theta_1x_n))^2}{n}
$$

Para o caso de LSM, não é necessário obter a _média_ dos quadrados. Apenas a soma dos mesmos já é suficiente como uma função de custo (_cost function_). Sendo $J(\theta_0, \theta_1)$ a função de custo: 

$$
J(\theta_0, \theta_1) = \sum_{i=1}^{n}(y_{real} - (\theta_0 + \theta_1x_n))^2
$$

#### Relembrando multiplicação de matrizes

Para expressarmos a função acima no formato vetorial, precisamos lembrar de como funciona a multiplicação de matrizes. Considere as matrizes $A$ e $B$:

$$
A = \begin{pmatrix}
a_{11} & a_{12} & a_{13}\\
a_{21} & a_{22} & a_{23}
\end{pmatrix}
B = 
\begin{pmatrix}
b_{11} & b_{12}\\ 
b_{21} & b_{22}\\ 
b_{31} & b_{32}
\end{pmatrix}
$$

A multiplicação de matrizes (conforme explicado [aqui](https://brasilescola.uol.com.br/matematica/multiplicacao-matrizes.htm)) é feita Linha x Coluna. Considerando a matriz resposta $C$, teríamos:

$c_{11} = a_{11}.b_{11} + a_{12}.b_{21} + a_{13}.b_{31}$

$c_{12} = a_{11}.b_{12} + a_{12}.b_{22} + a_{13}.b_{32}$

$c_{21} = a_{21}.b_{11} + a_{22}.b_{21} + a_{23}.b_{31}$

$c_{22} = a_{21}.b_{12} + a_{22}.b_{22} + a_{23}.b_{32}$

Com isso, teríamos:

$$
C = 
\begin{pmatrix}
c_{11} & c_{12}\\ 
c_{21} & c_{22}
\end{pmatrix}
$$

Essa propriedade da multiplicação acontecer linhas x colunas é muito importante para entendermos todas as vezes que a transposição da matriz for usada nos cálculos vetoriais para ML.

#### Aplicando a Notação Vetorial

Como queremos manter as relações $x_1.\theta_1, x_2.\theta_2, ..., x_n.\theta_n$, precisamos que as respectivas matrizes sejam multiplicadas na ordem correta. Para tal, realizamos a _transposição_ de uma das matrizes. A transposição nada mais é do que a transformação das linhas em colunas e colunas em linhas.

Vamos voltar à nossa equação de custo:

$$
J(\theta_0, \theta_1) = \sum_{i=1}^{n}(y_{real} - (\theta_0 + \theta_1x_n))^2
$$

Quando transformamos ela para uma notação vetorial, ficaria:

$$
J(\theta) = (y - \theta\cdot X)^2
$$

Note, porém, que o quadrado aqui é a multiplicação da matriz por ela mesma. Contudo, precisamos manter as posições dos vetores para a multiplicação. Não podemos fazer uma multiplicação de matrizes direta aqui. Dessa forma, a equação vetorial é, na verdade:

$$
J(\theta) = (y - X\cdot\theta)^T\cdot(y - X\cdot\theta)
$$

#### Minimizando a função custo

A função $J(\theta)$ é a função que precisamos minimizar: a soma dos quadrados das diferenças precisa ser a menor possível. Para descobrir o valor mínimo dessa função é necessário recorrer à sua derivada. Lembre que a derivada mostra a taxa de variação (ou a inclinação da reta tangente) de uma função em um ponto.

Com isso, o menor valor possível para essa função é encontrado quando a sua deriavada é igual a zero. Sem entrar em detalhes do processo, veja abaixo a derivada da nossa função de custo:

$$
\frac{\partial J}{\partial\theta} = -2X^T\cdot (y - X\cdot\theta) = 0
$$

Vamos agora abaixo deixar a fórmula em termos de $\theta$, dado que o resultado da derivada é $0$.

$-2X^T\cdot (y - X\cdot\theta) = 0$

$-2X^Ty + 2X^T X\cdot\theta = 0$

$2X^T X\cdot\theta = 2X^Ty$

$X^T X\cdot\theta = X^Ty$

$\theta = \frac{X^Ty}{X^T X}$

$\theta = (X^T\cdot X)^{-1}\cdot X^T\cdot y$

> Da mesma forma que a multiplicação, a divisão de matrizes não pode ser feita diretamente de forma que $\frac{A}{A}=1$. Também precisamos lembrar que $\frac{A}{B} = A\cdot B^{-1}$. Para saber mais sobre o cálculo do inverso de uma matriz, [clique aqui](https://www.todamateria.com.br/matriz-inversa/).

Chegamos, então, na nossa _equação normal_ para a função de custo. Essa é a função que será utilizada no código da regressão linear.

### Implementando em código

Agora que chegamos na equação normal que minimiza nossa função de custo, basta encontrarmos ela para os valores do dataset. Em python, isso pode ser implementado da seguinte forma:

```python
import numpy as np

def fit_lsm(self, x: np.array, y: np.array) -> np.array:
    size = x.shape[0]
    x_bias = np.c_[np.ones((size, 1)), x]
    theta_param = np.linalg.inv(x_bias.T.dot(x_bias)).dot(x_bias.T).dot(y) 
    return theta_param
```

* `x: np.array` é a matriz com os atributos do dataset
* `y: np.array` é o array com valores alvo
* `x_bias = np.c_[np.ones((size, 1)), x]` adiciona o atributo referente a $\theta_0$. Lembre que temos um coeficiente $\theta$ para cada atributo de $X$ e um parâmetro de bias. Esse parâmetro é incluído com o valor neutro para multiplicação ($1$) para permitir que seja considerado no vetor de $\theta$.
* `np.c_` função do `numpy` para a concatenação de vetores
* `np.linalg.inv` função do `numpy` para obter a inversa de uma matriz, equivalente ao $A^{-1}$
* `.T` é a transversa de uma matriz
* `.dot` função do `numpy` para multiplicação de matrizes

O valor resultante é um vetor que pode ser multiplicado com qualquer nova coordenada para obter a predição do seu valor alvo ($y$).