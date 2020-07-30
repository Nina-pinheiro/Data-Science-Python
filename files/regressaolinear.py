# Modelo de Regressão Linear: Encontra relações entre duas ou mais variáveis , o modelo ajusta uma equação linear entre os dados

# Objective: Estimar o lucro de empresas para direcionar os investimentos em empresas com maior potencial

# Importar bibliotecas usadas


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler


# Abrir o arquivo e visualizá-lo

data = pd.read_excel('investimento.xlsx')
print(data.head())

# Realizar a Análise descritiva dos dados

print(data.describe())

# Análise exploratória sobre as variávies
print(data.info())

# Verificar a correlação das variáveis
print(data.corr())


# Plotar o Gráfico de dispersão para maiores análises

figure = plt.figure(figsize=(9,6))
plt.scatter(data['Investimento'], data['lucro'])
plt.xlabel('lucro')
plt.ylabel('Investimento')


# Plotar o Gráfico de Boxplot e verificar  possiveis outliers

data.boxplot('Investimento')
data.boxplot('lucro')
plt.scatter(data.lucro, data.Investimento, c = "green", marker = "s")
plt.title("Verificar os possíveis outliers")
plt.xlabel("Investimento")
plt.ylabel("lucro")
plt.show()

# Processamento dos dados


scaler = StandardScaler()
data_scaled = scaler.fit_transform(data.drop('Empresa', axis=1))
columm_grafic = ['Investimento', 'lucro']
process_dados = pd.DataFrame(data_scaled, columns = columm_grafic)

print(process_dados.head())