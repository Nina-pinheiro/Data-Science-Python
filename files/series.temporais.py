# Importe das bibliotecas necessárias
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from scipy import stats
import datetime
from matplotlib.dates import DateFormatter
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import acf,pacf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima_model import ARIMA
from datetime import datetime, timedelta
from pandas import DataFrame

# Importar a base de dados
df = pd.read_csv('dataset/base.csv', header =0, sep = ',', decimal = ',')

# Análise descritiva

df.head()
df.describe()
df.info()

# Colocando volume como um número inteiro
df['volume'] = df['volume'].astype(np.int32)
# Colocando a coluna data com formato de data
df['data'] = pd.to_datetime(df['data'])

# Plotando o gráfico para ver como se comporta a série
df.plot(x='data')
plt.xlabel('data', fontsize = 15)
plt.ylabel('volume',fontsize = 15)
plt.grid()

# Verificando as funções ACF e PACF
acf = plot_acf(df['volume'], lags = 40)
pacf = plot_pacf(df['volume'], lags = 40)

# Verificando a Estacionariedade

test_estacionariedade = adfuller(df['volume'])
output = pd.Series(test_estacionariedade[0:4], index = ['Teste', 'p-valor','Lags','Número de observações usadas'])

for key, value in test_estacionariedade[4].items():
  output['Valor crítico(%s)' %key] = value
print(output)

# Diferenciações

df['primeira_dif'] = df['volume'].diff()
df.head()
df['primeira_dif'].plot()

# Verificando a estacionariedade da série diferenciada
test_estacionariedade_dif1 = adfuller(df['primeira_dif'].dropna())
output_dif = pd.Series(test_estacionariedade_dif1[0:4], index = ['Teste', 'p-valor','Lags','Número de observações usadas'])

for key, value in test_estacionariedade_dif1[4].items():
  output['Valor crítico(%s)' %key] = value
print(output_dif)


# Realizar uma diferenciação por log
df['diferenciacao_log'] = np.log(df['volume'])
df.head()
df['diferenciacao_log'].plot()
