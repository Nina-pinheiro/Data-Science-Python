# Importando bibliotecas necessárias
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns 
import warnings
from mpl_toolkits.mplot3d import Axes3D
from sklearn import preprocessing
from sklearn.cluster import KMeans
import plotly as py
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.metrics import make_scorer
from sklearn.metrics import accuracy_score, f1_score, precision_score, roc_auc_score, recall_score, confusion_matrix
import seaborn as sns
warnings.filterwarnings("ignore")
py.offline.init_notebook_mode(connected = True)
from sklearn.metrics import silhouette_score


# Carregar dataset
df = pd.read_csv('dataset/data.csv')

# Renomear o nome da coluna
df.rename(columns={'clientid':'Id'},inplace = True)

# Copiando um dataset e instanciando
df_feature = df.copy()
# Criação de uma nova feature
df_feature['renda_liquida'] = df_feature['income'] - df_feature['loan']

# Remoção dos valores nulos
df_feature.dropna(inplace = True)
# Arredondar os valores presentes na coluna age
df_feature['age'] = df_feature['age'].apply(lambda x: round(x))

# Remoção de valores que apresentam idade menor que 0

df_feature.drop(df.query('age < 0 ').index, inplace = True)
# Forma alternativa
#df = df.query('age > 0')

# Separando variáveis 

X = df_feature.drop(columns = ['default','Id'])
y = pd.DataFrame(df_feature['default'].values)

# Normalização dos dados

normalizar = MinMaxScaler()
X = pd.DataFrame(normalizar.fit_transform(X))
X


# Parâmetros n_jobs = -1 vai para o ultimo da lista
parametros = {'C': [0.1,1,10],
              'solver': ['liblinear'],
              'penalty':['l1','l2'],
              'max_iter':[10]}

# Modelo de Regressão logística
regressao = LogisticRegression()
cv = KFold(shuffle = True)

# Utilização do Grid search
grid = GridSearchCV(regressao,param_grid = parametros, n_jobs = -1, cv = cv)

# Separação de treino e teste
X_train,X_test,y_train,y_test = train_test_split(X, y,test_size = 0.2)
grid.fit(X_train, y_train)

lr = LogisticRegression(**grid.best_params_)
lr.fit(X_train, y_train)
lr.predict(X_test)

# Criação da Matriz de Confusão
sns.heatmap(confusion_matrix(y_test, lr.predict(X_test)), annot=True, fmt = 'd')