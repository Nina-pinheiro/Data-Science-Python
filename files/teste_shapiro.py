""" Projeto Nina
Python 3.7

Teste Shapiro é importante para verificar se os dados segue a distribuição normal, ou seja, Gaussiana.
 Se os dados segue essa distribuição, portanto os dados serão simétricos e a média = mediana = moda

"""

# Importar as Bibliotecas necessárias

import scipy.stats as stats
import pandas as pd
import matplotlib.pyplot as plt

# Leitura do dataset

data = pd.read_csv('dataset/consulta.csv')

# Visualizar brevemente os 5 primeiros dados da planilha

print(data.head())

# Resume brevemente as estatísticas dos dados

print(data.describe())

# Verifica quais são os tipos das variáveis

print(data.dtypes)

# Plotar um Histograma da variável Idade

# É importante a visualização deste gráfico, pois podemos analisar grandes assimetrias, descontinuidade de dados
# e também picos multimodais. Assim identificando que se é ou não  um padrão de distribuição normal.

data.hist(column = "Idade", bins = 10, color = "pink")

# Colocar um título para o histograma

plt.title("Histograma da variável Idade")

plt.show()

# Realizo o Teste Shapiro
# É importante entender que no Teste Shapiro
# A Hipótese Nula (H0) = Se o p-value > nível de significância , então a distribuição é normal
# A Hipótese Alternativa (H1) = Se o p value < nível de significância , então a distribuição não é normal

# Nível de significância = 0.05

shapiro_stat, shapiro_p_valor = stats.shapiro(data.Idade)
print(shapiro_p_valor)
print(shapiro_stat)

# Condicional que verifica se a distribuição é normal

if shapiro_p_valor > 0.05:
    print("Com 95% de confiança, os dados são similares a uma distribuição normal")
else:
    print("Com 95% de confiança, os dados não são similares a uma distribuição normal")


