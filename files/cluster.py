# Importando as Bibliotecas necessárias
import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns 
import warnings
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import MinMaxScaler
from sklearn import preprocessing
from sklearn.cluster import KMeans
import plotly as py
warnings.filterwarnings("ignore")
py.offline.init_notebook_mode(connected = True)

# Carregando a base de dados:

data = pd.read_csv("pokemon.csv")

# Verificar as 5 primeiras linhas

data.head()

# Verificar o volume dos dados

data.shape

# Análise descritiva

data.describe()

# Verificar os tipos de dados

data.dtypes


# Verificando os registros nulos

data.isnull().sum()

# Removendo a coluna Type 2, pois tem muito 0 e variáveis categoricas

data_remover = data.drop(columns=['Name','Type 2','Generation','Legendary','Type 1'])


# Verificar as 5 primeiras linhas
data_remover.head()

# Verificar a correlação entre as variáveis - Ajuda a definir a melhor técnica

sns.pairplot(data[['HP','Attack','Defense','Sp. Atk','Sp. Def','Speed']])
plt.show()

# Definindo um estilo para os gráficos:
plt.style.use('fivethirtyeight')


# Verificando as distribuição dos dados:

# Figsize é o tamanho em polegadas ( largura, altura)
plt.figure(1 , figsize = (15 , 6))
n = 0 
for x in [ 'HP' , 'Attack','Defense','Speed','Sp. Def','Sp. Atk']:
    n += 1
    plt.subplot(1 , 6 , n)
    plt.subplots_adjust(hspace =0.5 , wspace = 0.5)
    sns.distplot(data_remover[x] , bins = 25)
    plt.title('{} '.format(x))
plt.show()

# Gráfico do Total
sns.distplot(data_remover.Total)
plt.show()

# Normalizar as variáveis de 0 a 1
 min_max_scaler = preprocessing.MinMaxScaler()
 variaveis_minmax = min_max_scaler.fit_transform(data_remover)

# Descobrir a inertia

kmeans.inertia_

# Realizando o método do cotovelo Ebbow para descobrir o número de cluster
SSE = []
for cluster in range(1,11):
    kmeans = KMeans(n_clusters = cluster, init='k-means++')
    kmeans.fit(variaveis_normalizadas)
    SSE.append(kmeans.inertia_)

#Converte o conjunto do dataframe em um gráfico
frame = pd.DataFrame({'Cluster':range(1,11), 'SSE':SSE})
plt.figure(figsize=(12,6))
plt.plot(frame['Cluster'], frame['SSE'], marker='o')
plt.xlabel('Número  of clusters')
plt.ylabel('Inertia - SSE- Erro')

# Realizando o algoritmo k-means, escolhendo os parâmetros, k-means++ tem uma maior eficiência do que o parâmetro de aleatoriedade
kmeans = KMeans(n_clusters = 4, init = 'k-means++')
cluster = kmeans.fit_predict(variaveis_normalizadas)

# Centroides
C = kmeans.cluster_centers_

# Plotando o gráfico

plt.scatter(variaveis_normalizadas[:,2],variaveis_normalizadas[:,3], c = cluster, cmap = 'rainbow')
plt.scatter(C[:,2] ,C[:,3], color='black', label = 'Centroids')

plt.title('Pokemon Clusters and Centroids')
plt.xlabel('Ataque')
plt.ylabel('Defesa')
plt.legend()

plt.show()


# Plotando o gráfico 3D para 3 variáveis

fig = plt.figure()
ax = Axes3D(fig)
ax.scatter(variaveis_normalizadas[:, 0], variaveis_normalizadas[:, 1], variaveis_normalizadas[:, 2])
ax.scatter(C[:, 0], C[:, 1], C[:, 2], marker='*', c='#050505', s=1000)

