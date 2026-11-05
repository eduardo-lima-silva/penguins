# Tratamento e Imputação de Dados Ausentes: Penguins

Dados reais quase sempre chegam com buracos. Este projeto simula um cenário difícil, com **30% das células apagadas aleatoriamente**, e reconstrói a base com técnicas de imputação escolhidas **de acordo com o tipo de cada variável**, verificando se a estrutura dos dados foi preservada.

> 🇺🇸 *Missing-data treatment on the Seaborn Penguins dataset: 30% of cells removed at random, EDA, z-score outlier removal and type-aware imputation (order-based fill + KNN Imputer).*

## Por que isso importa

Apagar toda linha com valor faltante (`dropna`) jogaria fora quase toda a base. Com 30% das células vazias, **praticamente nenhuma linha fica completa**. Já preencher tudo com a média distorce as distribuições e apaga as diferenças entre grupos.
Escolher bem a imputação é o que separa um modelo treinado em dados plausíveis de um modelo treinado em ruído.

## Dados

- Base **Penguins** (biblioteca seaborn): **344 pinguins**, 7 atributos (espécie, ilha, comprimento e profundidade do bico, comprimento da nadadeira, massa corporal e sexo).
- Com `seed = 42`, **723 das 2.408 células (30%)** são apagadas aleatoriamente, o que torna o experimento reproduzível.

## Etapas

1. **Análise exploratória:** pairplot por espécie, boxplots (massa, bico e nadadeira × espécie) e mapa de correlação.
   - A espécie *Gentoo* é mais pesada e tem nadadeiras maiores, e a *Chinstrap* tem o bico mais longo.
   - A nadadeira e a massa são fortemente correlacionadas. Na base completa, a regressão linear dá `nadadeira ≈ 0,0152 · massa + 137`.
2. **Outliers:** remoção por **z-score** (limiares de 2,6 e 3) no comprimento e na profundidade do bico, na base de referência.
3. **Variáveis categóricas (espécie e ilha):** a base está ordenada por espécie, então os vazios foram preenchidos com o **valor do registro anterior**. Para a ilha, antes de preencher, verifiquei com médias por espécie e ilha que **a ilha não diferencia as medidas físicas**, só a espécie.
4. **Variáveis numéricas e sexo:** normalização **Min-Max** seguida de **`KNNImputer` (k = 5, ponderado pela distância)**. O sexo foi codificado como 0/1, imputado e arredondado de volta para Male/Female.
5. **Registros sem informação suficiente:** as **13 linhas com 4 ou mais atributos numéricos vazios** foram descartadas em vez de "inventadas", e a base final ficou com **331 registros completos**.
6. **Validação visual:** pairplot e gráfico 3D (Plotly) com z-score antes e depois da imputação, para conferir que **a separação entre as espécies se manteve**.

## Resultado

Uma base completa (331 × 7) em que os valores imputados respeitam as relações entre as variáveis. O KNN usa os pinguins mais parecidos, e não a média geral, então os grupos de espécies continuam separados nos gráficos.

## Limitações e próximos passos

- O preenchimento pela ordem dos registros só funciona porque a base está ordenada por espécie. Para a ilha, ele pode errar entre pinguins *Adelie*, que vivem em três ilhas.
- A qualidade foi checada visualmente. Um próximo passo é **medir o erro de imputação** (RMSE e acurácia do sexo) comparando com os valores originais apagados, e comparar o KNN com média, mediana e `IterativeImputer`.

## Como executar

```bash
pip install pandas numpy scipy seaborn matplotlib plotly scikit-learn jupyter
jupyter notebook penguins.ipynb
```

A base é baixada automaticamente pelo seaborn (`sns.load_dataset("penguins")`).

## Tecnologias

Python · pandas · NumPy · SciPy · scikit-learn (KNNImputer, MinMaxScaler) · seaborn · matplotlib · Plotly
