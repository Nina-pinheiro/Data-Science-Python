# Importar as bibliotecas necessárias

import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import seaborn as sns
from sklearn.linear_model import LinearRegression

# Leitura do dataset

df = pd.read_csv("dataset/consumo.csv") 

# Converter uma coluna para numerica

df['Temperatura Maxima (C)'] = df['Temperatura Maxima (C)'].str.replace(',','.').astype(float)
df['Temperatura Minima (C)'] = df['Temperatura Minima (C)'].str.replace(',','.').astype(float)
df['Precipitacao (mm)'] = df['Precipitacao (mm)'].str.replace(',','.').astype(float)
df['Temperatura Media (C)'] = df['Temperatura Media (C)'].str.replace(',','.').astype(float)

# Análise descritiva

df.describe()
df.head()
df.dtypes
df.info()
df.tail()
df.shape

# Verificar quais são os valores faltantes

df.isnull().sum()

# Remover todos os valores faltantes
df.dropna(how = "all", inplace = True)

# Copiando um data frame em uma nova variável 

df_feature = df.copy()

# Criação de uma nova feature

df_feature['variacao'] = (df_feature['Temperatura Maxima (C)']) - (df_feature['Temperatura Minima (C)'])
df_feature

# Plotando o gráfico da nova feature
df_feature.plot(x='variacao', y = 'Consumo de cerveja (litros)')
plt.xlabel('variacao', fontsize = 15)
plt.ylabel('Consumo de cerveja (litros)',fontsize = 15)
plt.grid()

# Excluindo a coluna data
df_feature = df_feature.drop(columns = 'Data')

# Realizar a matriz de correlação

df_feature.corr().round(3)

# Gráficos

plt.figure()
sns.pairplot(df_feature,x_vars=['Temperatura Minima (C)','Temperatura Media (C)','Temperatura Maxima (C)','Precipitacao (mm)','variacao'],
             y_vars=['Consumo de cerveja (litros)'],hue='Final de Semana',diag_kind=None)

# Realizar o gráfico de final de semana e consumo de cerveja
plt.figure(2)
sns.swarmplot(x='Final de Semana',y='Consumo de cerveja (litros)',data= df_feature)
plt.grid()
plt.xlabel('Final de semana')
plt.ylabel('Consumo de cerveja [L]')

# Realizar o gráfico de final de semana e variacao(nova feature criada)

plt.figure(3)
sns.swarmplot(x = 'Final de Semana', y = 'variacao', data = df_feature)
plt.grid()
plt.xlabel('Final de semana')
plt.ylabel('variacao')


# Utilizando o modelo de regressão linear
modelo = LinearRegression()

# Colocando a variável target
y = df_feature['Consumo de cerveja (litros)'].values #target

# colocando as variaveis independentes neste exemplo pega todos menos consumo de cerveja
x = df_feature.drop(columns='Consumo de cerveja (litros)').values #fetures
xColunas = df_feature.drop(columns='Consumo de cerveja (litros)').columns

# Realizando o treinamento 

xTrain,xTest,yTrain,yTest = train_test_split(x,y, test_size = 0.3, random_state = 54564541)

# Fitando o modelo

modelo.fit(xTrain,yTrain)
yPred = modelo.predict(xTest)

# Calcular os resíduos

res = yPred - yTest

# Testes

print('Valor de R2: {}'.format(modelo.score(xTest,yTest)))
print('Valor MSE: {}' .format(mean_squared_error(yTest,yPred)))
print('Coeficientes da regressão: {}'.format(modelo.coef_))
print('Intercept da regressão: {} \n'.format(modelo.intercept_))

