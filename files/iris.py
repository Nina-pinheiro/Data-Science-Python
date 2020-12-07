"""Nome: Nina

Projeto utilizando a database iris que é do pacote sklearn 



Tarefa 4 - Testando os métodos de agrupamento com conjunto de dados íris. 
Tarefas computacionais • 10 pontos O conjunto de dados de flores de íris foi apresentado pelo estatístico e biólogo britânico
Ronald Fisher em seu artigo de 1936 "O uso de medições múltiplas em problemas taxonômicos". 
Este é talvez o banco de dados mais conhecido que pode ser encontrado na literatura de reconhecimento de padrões.
O conjunto de dados de íris fornece as medidas em centímetros das variáveis comprimento e largura da sépala e comprimento e 
largura da pétala, respectivamente, para 50 flores de cada uma das 3 espécies de íris. As espécies são íris setosa, versicolor e
virginica. O objetivo desta tarefa é aprender utilizar vários métodos de agrupamento sobre os dados de íris e avaliação de 
qualidade de agrupamento utilizando os rótulos de classe fornecidos nos dados. Para carregar os dados de íris podemos utilizar a 
função fornecida pelo pacote SciKit Learn: from sklearn.datasets import load_iris data = load_iris() .
Usando estes dados realizaremos a comparação de métodos de agrupamento de k-Means, hierárquico e por densidade.

1.Carrega os dados de atributos usando a função acima. Coloca-os numa estrutura DataFrame do pacote Pandas e mostra as estatisticas de atributos (função df.describe()).
2.Pré-processamento de dados realizando a normalização dos atributos.
3.Realize agrupamento k-Means (usando sklearn.cluster.KMeans) supondo 2 grupos e pontos centrais inicias aleatorios. Discute como melhorar a escolha dos pontos centrais iniciais. Faça um agrupamento supondo 3 grupos. Calcule a coesão e separação de agrupamentos em 2 e 3 grupos e determine qual número de grupos é melhor.
4.Realize um agrupamento hierárquico (usando sklearn.cluster.AgglomerativeClustering) e produze o dendrograma dele.
5.Realize um agrupamento por densidade (sklearn.cluster.DBSCAN). Teste como agrupamento depende dos parámetros eps e min_samples. Informe sobre o número estimado de clusters e pontos de ruído e outros parametros de avaliação de agrupamento de forma semelhante a exemplo https://scikit-learn.org/stable/auto_examples/cluster/plot_dbscan.html#sphx-glr-download-auto-examples-cluster-plot-dbscan-py.
6.Realizar a avaliação dos métodos de agrupamentos testados. Material complementar online: - SciPy Hierarchical Clustering and Dendrogram Tutorial https://joernhees.de/blog/2015/08/26/scipy-hierarchical-clustering-and-dendrogram-tutorial/ - Hierarchical Clustering - Dendrograms Using Scipy and Scikit-learn in Python https://www.youtube.com/watch?v=JcfIeaGzF8A


"""

# Importando Bibliotecas

from sklearn import datasets
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn import preprocessing
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram
from matplotlib import pyplot as plt

#1.
# Carregar os dados

iris = datasets.load_iris()

# Carregar no Dataframe

df = pd.DataFrame(data= np.c_[iris['data'], iris['target']],
                     columns= iris['feature_names'] + ['target'])

# Realizar as análises descritivas

df.describe()

#2.

# Pré-processamento de dados realizando a normalização dos atributos- Minimo Maximo

# Normalizar as variáveis de 0 a 1
min_max_scaler = preprocessing.MinMaxScaler()
variaveis_normalizadas = min_max_scaler.fit_transform(df)


#3 Realize agrupamento k-Means (usando sklearn.cluster.KMeans) supondo 2 grupos e pontos centrais inicias aleatorios. Discute como melhorar a escolha dos pontos centrais iniciais. 
#Faça um agrupamento supondo 3 grupos. Calcule a coesão e separação de agrupamentos em 2 e 3 grupos e determine qual número de grupos é melhor.

kmeans2 = KMeans(n_clusters = 2, init = 'k-means++')
cluster2 = kmeans2.fit_predict(variaveis_normalizadas)

kmeans3 = KMeans(n_clusters = 3, init = 'k-means++')
cluster3 = kmeans3.fit_predict(variaveis_normalizadas)

# Falta expllicar pq o kmeans++ é melhor que o aleatorio no site explica melhor do sktleranin
# coesao e separação falta calcular


# 4Realize um agrupamento hierárquico (usando sklearn.cluster.AgglomerativeClustering) e produze o dendrograma dele

def plot_dendrogram(model, **kwargs):
    # Create linkage matrix and then plot the dendrogram

    # create the counts of samples under each node
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack([model.children_, model.distances_,
                                      counts]).astype(float)

    # Plotar o dendograma
    dendrogram(linkage_matrix, **kwargs)




# setting distance_threshold=0 ensures we compute the full tree.
model = AgglomerativeClustering(distance_threshold=0, n_clusters=None)

model = model.fit(df)
plt.title('Dendograma')
# plot the top three levels of the dendrogram
plot_dendrogram(model, truncate_mode='level', p=3)
plt.xlabel("Observações")
plt.show()