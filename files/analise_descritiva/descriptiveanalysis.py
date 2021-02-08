""" Project DataScience in Python
Developed: Nina Machado

Python 3.7

"""

# Required Libraries

import matplotlib.pyplot as plt
import pandas as pd
import statsmodels.api as sm
import pylab as pl
import numpy as np
from patsy import dmatrix  

# Read the data in dataset
  
df = pd.read_csv('dataset/consulta.csv')  

# View the first 5 datasets of the spreadsheet

print (df.head())

# Summary of descriptive data analysis

print ('Data Summary') 
print (df.describe() )

# Function that calculates the standard deviation of the numeric columns

print (df.std() )
 
# Creates a histogram chart and plot all of the columns

df.hist()
pl.show()

