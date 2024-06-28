# Random Forest

[EM DESENVOLVIMENTO]

 - Coleção de árvores não relacionadas

 ## Bootstrap

 Uma técnica de estatística de amostragem, onde elementos são selecionados aleatoriamente, mas com reposição. Isso significa que um elemento pode ser escolhido mais de uma vez e que alguns elementos podem nunca serem escolhidos.

 ## Treinamento das árvores

 Cada árvore é treinada com uma amostra realizada com Bootstrap. No entanto, não são escolhidas todas as features em todas as árvores, mas são escolhidas também por amostragem aleatória, tentando evitar o overfitting ou a existência de alguma feature que gere bons resultados em entropia e gini mas não tenha real ganho de informação (como um id, por exemplo).

 ## Classificação

 É realizada uma votação pelo voto majoritário de todas as árvores.

 // a quantidade de votos não poderia implicar a porcentagem de chance de ser? Segundo o chatgpt sim e até existe no sklearn a propriedade predict_proba.