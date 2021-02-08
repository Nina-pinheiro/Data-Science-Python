# Objetivos: aprender a carregar arquivos .csv ou excel do computador utilizando pacote Pandas.


# Importar Bibliotecas necessárias

import pandas as pd

# Abrir o arquivo e visualizá-lo - Extensão XLSX

data = pd.read_excel('dataset/investimento.xlsx')
# Comando head mostra as primeiras linhas do dataframe
print(data.head())

# Abrir o arquivo e visualizá-lo -  Extensão CSV

data = pd.read_csv('dataset/consulta.csv')
print(data.head())

# Transformando um dicionário em dataframe(Dataframe tem o mesmo comprimento)

# Dicionário - chave e valor

pessoa = {'Filmes': ['A insustentável leveza do ser', 'Matrix','Laranja mecânica'],
          'Animais':['cachorro','coelho','gato'],
          'Cores': ['Azul','Rosa','Verde'],
          'Viagens': ['Peru','Tailandia','Portugal']}

dataframe = pd.DataFrame(pessoa)

# Utilizando o comando Series - apresenta características unidimensional com indices

series = pd.Series([2,5,4,3])
print(series)


# Utilizar comando describe que calcula média, desvio padrão, percentis e etc

analise_alunos = {'Nomes':['Daniela','Ricardo','Joaquim','Alice'],
                  'Notas':['10','5','8','9'],
                  'Aprovado':['Sim','Não','Sim','Não'],
                  'Sexo':['F','M','M','F']}

dataframe_analise = pd.DataFrame(analise_alunos)

dataframe_analise.head()

# Quantas linhas e quantas colunas tem o dataframe

dataframe_analise.shape

# Ter uma visão geral do dataframe usando a função describe

dataframe_analise.describe()

# Fatiamento de colunas 

dataframe_analise['Nomes']

# Fatiamente de linhas no caso a primeira e a segunda

dataframe_analise.loc[[0,1]]

# Fatiamento de linhas colocando numa lista lendo da primeira linha até a terceira

dataframe_analise.loc[0:2]

# Pegar uma linha de uma coluna específica

dataframe_analise.loc[dataframe_analise['Nomes']== 'Alice']





