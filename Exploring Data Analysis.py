# -*- coding: utf-8 -*-
"""
Created on Sun Jun  4 16:19:18 2023

@author: swank
"""


Defining Descriptive Statistics for Numeric Data
%matplotlib inline

#iris dataset is depricated
from sklearn.datasets import load_iris
iris = load_iris()
import pandas as pd
import numpy as np
​#Unicode error U+200B = specific symbol not allowed by unicode

print('Your pandas version is: %s' % pd.__version__)
print('Your NumPy version is %s' % np.__version__)
from sklearn.datasets import load_iris
iris = load_iris()
iris_nparray = iris.data
​
iris_dataframe = pd.DataFrame(iris.data, columns=iris.feature_names)
iris_dataframe['group'] = pd.Series([iris.target_names[k] for k in iris.target], dtype="category") 


#Measuring central tendency
print(iris_dataframe.mean(numeric_only=True))
print(iris_dataframe.median(numeric_only=True))

#Measuring variance and range
print(iris_dataframe.std())
print(iris_dataframe.max(numeric_only=True) 
      - iris_dataframe.min(numeric_only=True))

#Working with percentiles
print(iris_dataframe.quantile([0,.25,.50,.75,1]))

#Defining measures of normality
from scipy.stats import kurtosis, kurtosistest
variable = iris_dataframe['petal length (cm)']
k = kurtosis(variable)
zscore, pvalue = kurtosistest(variable)
print('Kurtosis %0.3f z-score %0.3f p-value %0.3f' 
      % (k, zscore, pvalue))

from scipy.stats import skew, skewtest
variable = iris_dataframe['petal length (cm)']
s = skew(variable)
zscore, pvalue = skewtest(variable)
print('Skewness %0.3f z-score %0.3f p-value %0.3f' 
      % (s, zscore, pvalue))

#Counting for Categorical Data
pcts = [0, .25, .5, .75, 1]
iris_binned = pd.concat(
    [pd.qcut(iris_dataframe.iloc[:,0], pcts, precision=1),
     pd.qcut(iris_dataframe.iloc[:,1], pcts, precision=1),
     pd.qcut(iris_dataframe.iloc[:,2], pcts, precision=1),
     pd.qcut(iris_dataframe.iloc[:,3], pcts, precision=1)],
    join='outer', axis = 1)

#Understanding frequencies
print(iris_dataframe['group'].value_counts())
print(iris_binned['petal length (cm)'].value_counts())
print(iris_binned.describe())

#Creating contingency tables
print(pd.crosstab(iris_dataframe['group'],
                  iris_binned['petal length (cm)']))
Creating Applied Visualization for EDA
Inspecting boxplots
boxplots = iris_dataframe.boxplot(fontsize=9)
import matplotlib.pyplot as plt
boxplots = iris_dataframe.boxplot(column='petal length (cm)', 
                                  by='group', fontsize=10)
plt.suptitle("")
plt.show()

#Performing t-tests after boxplots
from scipy.stats import ttest_ind
​
group0 = iris_dataframe['group'] == 'setosa'
group1 = iris_dataframe['group'] == 'versicolor'
group2 = iris_dataframe['group'] == 'virginica'
variable = iris_dataframe['petal length (cm)']
​
print('var1 %0.3f var2 %03f' % (variable[group1].var(), 
                                variable[group2].var()))
variable = iris_dataframe['sepal width (cm)']
t, pvalue = ttest_ind(variable[group1], variable[group2],
                      axis=0, equal_var=False)
print('t statistic %0.3f p-value %0.3f' % (t, pvalue))
from scipy.stats import f_oneway
variable = iris_dataframe['sepal width (cm)']
f, pvalue = f_oneway(variable[group0], 
                     variable[group1], 
                     variable[group2])
print('One-way ANOVA F-value %0.3f p-value %0.3f' 
      % (f,pvalue))

#Observing parallel coordinates
from pandas.plotting import parallel_coordinates
iris_dataframe['group'] = iris.target
iris_dataframe['labels'] = [iris.target_names[k] 
                    for k in iris_dataframe['group']]
pll = parallel_coordinates(iris_dataframe, 'labels')

#Graphing distributions
cols = iris_dataframe.columns[:4]
densityplot = iris_dataframe[cols].plot(kind='density')
variable = iris_dataframe['petal length (cm)']
single_distribution = variable.plot(kind='hist')

#Plotting scatterplots
palette = {0: 'red', 1: 'yellow', 2:'blue'}
colors = [palette[c] for c in iris_dataframe['group']]
simple_scatterplot = iris_dataframe.plot(
                kind='scatter', x='petal length (cm)', 
                y='petal width (cm)', c=colors)
from pandas.plotting import scatter_matrix
palette = {0: "red", 1: "yellow", 2: "blue"}
colors = [palette[c] for c in iris_dataframe['group']]
matrix_of_scatterplots = scatter_matrix(
    iris_dataframe, figsize=(6, 6), 
    color=colors, diagonal='kde')

#Understanding Correlation
Using covariance and correlation
iris_dataframe.cov()
iris_dataframe.corr()
covariance_matrix = np.cov(iris_nparray, rowvar=0)
correlation_matrix = np.corrcoef(iris_nparray, rowvar=0)

#Using nonparametric correlation
from scipy.stats import spearmanr
from scipy.stats.stats import pearsonr
a = iris_dataframe['sepal length (cm)']
b = iris_dataframe['sepal width (cm)']
rho_coef, rho_p = spearmanr(a, b)
r_coef, r_p = pearsonr(a, b)
print('Pearson r %0.3f | Spearman rho %0.3f' 
      % (r_coef, rho_coef))

#Considering chi-square for tables
from scipy.stats import chi2_contingency
table = pd.crosstab(iris_dataframe['group'], 
                    iris_binned['petal length (cm)'])
chi2, p, dof, expected = chi2_contingency(table.values)
print('Chi-square %0.2f p-value %0.3f' % (chi2, p))

#Modifying Data Distribution
#Creating a Z-score standardization
from sklearn.preprocessing import scale
variable = iris_dataframe['sepal width (cm)']
stand_sepal_width = scale(variable)

#Transforming other notable distributions
from scipy.stats.stats import pearsonr
tranformations = {'x': lambda x: x, 
                  '1/x': lambda x: 1/x, 
                  'x**2': lambda x: x**2, 
                  'x**3': lambda x: x**3, 
                  'log(x)': lambda x: np.log(x)}
a = iris_dataframe['sepal length (cm)']
b = iris_dataframe['sepal width (cm)']
for transformation in tranformations:
    b_transformed =  tranformations[transformation](b)
    pearsonr_coef, pearsonr_p = pearsonr(a, b_transformed)
    print('Transformation: %s \t Pearson\'s r: %0.3f' 
          % (transformation, pearsonr_coef))