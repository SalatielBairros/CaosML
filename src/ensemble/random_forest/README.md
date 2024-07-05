# Random Forest

## Introdução

Random Forest é um algoritmo de _ensemble learning_ que utiliza múltiplas árvores de decisão, ou seja, uma coleção de árvores não relacionadas, para melhorar a precisão e robustez das previsões. Cada árvore é construída a partir de uma amostra aleatória dos dados e a decisão final é tomada com base na votação ou média das previsões das árvores individuais.

Algoritmos de _ensemble_ realizam o treinamento de múltiplos modelos "fracos" que, combinados, criam um modelo "forte", capaz de trazer resultados que compensem os erros típicos dos modelos utilizados. Por exemplo, árvores de decisão são muito suscetíveis a _overfitting_, adicionando viéses nos seus resultados. A combinação de difersos modelos de árvores de decisão utilizando uma técnica de seleção aleatória dos dados chamada _bootstrap_ ajuda a reduzir consideravelmente esse problema. 

## Implementação

### Bagging e Bootstrap

O Bootstrap é uma técnica de estatística de amostragem, onde elementos são selecionados aleatoriamente, mas com reposição. Isso significa que um elemento pode ser escolhido mais de uma vez e que alguns elementos podem nunca serem escolhidos.

Isso significa que cada _estimator_, ou seja, cada árvore de decisão, recebe como input para treinamento uma amostra de dados que pode conter dados repetidos e também não contém alguns dos dados originais. Da mesma forma, não são todas as features que são enviadas para cada árvore. Esse uso de várias amostrar para diferentes modelos é chamada de _Bagging_.
 
> A porcentagem das features que são selecionadas aleatoriamente e enviadas para cada árvore é configurada no parâmetro `max_features`.

O algoritmo foi implementado da forma abaixo. Note que ele não trabalha diretamente com os dados, mas com os índices para a seleção. Não é necessário computacionalmente passar pelos itens em si para selecioná-los, basta selecionar índices aleatórios dentro do tamanho dos dados enviados para treino. No caso implementado abaixo, o número de registros que cada árvore recebe é do mesmo tamanho do input original, com a diferença de conter dados repetidos e deixar outros de fora.

```python
def __bootstrap__(self, training_data_size: int, training_data_n_features: int):
    # cria um array com a lista de todos os índices baseados no tamanho do dado de treinamento
    sample_indexes = list(range(training_data_size))

    # seleciona aleatoriamente e com repetição os índices que serão utilizados
    selected_rows_indexes = np.random.choice(sample_indexes, training_data_size)

    # define o número de features a ser selecionadas baseadas no parâmetro configurado
    n_features_to_select = int(training_data_n_features * self.max_features)

    # seleciona aleatoriamente mas sem repetição as features a serem utilizadas
    selected_columns_indexes = np.random.permutation(training_data_n_features)[:n_features_to_select]

    return selected_rows_indexes, selected_columns_indexes
```

O método acima é executado para cada um dos modelos implementados. O número total de modelos é configurado pelo parâmetro `n_estimators`. A implementação do bootstrap para todos os `estimators` pode ser vista no método `__bootstrap_all_estimators__`.

### Treinamento das árvores

Cada árvore é treinada com uma amostra realizada com Bootstrap. No entanto, não são escolhidas todas as features em todas as árvores, mas são escolhidas também por amostragem aleatória. Isso é feito porque é possível que exista uma propriedade que gere ótimos valores de entropia ou gini, mas não traga um ganho real de informação. Um exemplo disso seria uma feature que contenha um identificador único para cada linha. Sempre que essa coluna fosse dividida, existiria um suposto ganho de informação e, quanto maior a sua fragmentação, maior seria o suposto ganho. No entanto, sabemos que isso não é verdade e que um identificador único não traz nenhuma informação para um algoritimo de classificação ou regressão. Por isso que há uma seleção aleatória de features, para que uma coluna não se sobreponha às outras como decisória no resultado do modelo.

A implementação do treinamento é bastante simples. A maior complexidade está realmente na seleção dos dados para cada modelo do ensemble.

```python
def fit(self, X: np.array, y:np.array):
    training_data_size, training_data_n_features = X.shape
    self.__bootstrap_all_estimators__(training_data_size, training_data_n_features)
    for estimator in self.estimators:
        estimator.fit(X, y)
```

### Out of Bag score
O Out-of-Bag (OOB) Score é uma técnica de validação interna utilizada em algoritmos de ensemble baseados em bagging, como o Random Forest. Durante o treinamento, cada árvore de decisão é construída usando uma amostra aleatória do conjunto de dados com reposição, deixando de fora, uma parte dos dados, conhecidos como "out-of-bag". O OOB Score é calculado avaliando o desempenho do modelo em prever esses dados deixados de fora, fornecendo uma estimativa de erro que é equivalente à validação cruzada, mas sem a necessidade de um conjunto de validação separado. Essa abordagem é vantajosa porque permite a avaliação da performance do modelo de forma eficiente, utilizando todos os dados disponíveis tanto para treinamento quanto para validação, garantindo uma melhor utilização dos dados e uma estimativa mais confiável do desempenho do modelo em dados não vistos.

> A implementação desse score pode ser vista no método `out_of_bag_score`.

## Classificação

A classificação (ou predição) é o resultado do modelo. Também pode ser um valor de regressão, caso a implementação envolva árvores de regressão. Abaixo podemos ver a implementação do método `predict`. Note que o método realiza a predição em todos os modelos e, posteriormente, para cada registro (usando `predictions.T`), todas as predições realizadas para ele são votadas e o valor mais comum é selecionado. O modelo também retorna a confiabilidade da predição baseado na votação.

```python
def predict(self, X: np.array):
        predictions = []
        response = []
        size = X.shape[0]

        for estimator in self.estimators:
            prediction = estimator.predict(X)
            predictions.append(prediction)

        for estimator_prediction in np.array(predictions).T:
            counter = Counter(estimator_prediction)            
            prediction, frequency = counter.most_common(1)[0]
            prob = frequency / size
            response.append((prediction, prob))

        return response
```

## Conclusão e melhorias futuras

O modelo implementado mostra um pouco do poder dos métodos ensemble. Com uma implementação razoavelmente fácil, é possível melhorar consideravelmente a performance de apenas uma árvore de decisão. Sobre a implementação em si, algumas melhorias:

- Implementar o modelo de regressão
- Implementar testes unitários
- Permitir a recepção de critérios de avaliação customizados