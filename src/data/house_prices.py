import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy import stats
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

#%% load dataset
df = pd.read_csv('../../data/raw/house_prices.csv')
print(df.head())
print(df.columns)
print(df.shape)

#%% descriptive statistics summary
print(df['SalePrice'].describe())

#%% histogram
sns.distplot(df['SalePrice'])
plt.savefig('../../reports/figures/house_prices_saleprice_histogram.png')
plt.show()

#%% skewness and kurtosis
print('Skewness:', df['SalePrice'].skew())
print('Kurtosis:', df['SalePrice'].kurt())

#%% relationship with numeric variables
# scatter plot grlivarea/saleprice
var = 'GrLivArea'
data = pd.concat([df['SalePrice'], df[var]], axis=1)
data.plot.scatter(x=var, y='SalePrice', ylim=(0, 800000))
plt.savefig('../../reports/figures/house_prices_grlivarea_scatter_plot.png')
plt.show()

# scatter plot totalbsmtsf/saleprice
var = 'TotalBsmtSF'
data = pd.concat([df['SalePrice'], df[var]], axis=1)
data.plot.scatter(x=var, y='SalePrice', ylim=(0, 800000))
plt.savefig('../../reports/figures/house_prices_totalbsmtsf_scatter_plot.png')
plt.show()

#%% relationship with categorical variables
# box plot overallqual/saleprice
var = 'OverallQual'
data = pd.concat([df['SalePrice'], df[var]], axis=1)
f, ax = plt.subplots(figsize=(8, 6))
fig = sns.boxplot(x=var, y='SalePrice', data=data)
fig.axis(ymin=0, ymax=800000)
plt.savefig('../../reports/figures/house_prices_overallqual_box_plot.png')
plt.show()

# box plot yearbuilt/saleprice
var = 'YearBuilt'
data = pd.concat([df['SalePrice'], df[var]], axis=1)
f, ax = plt.subplots(figsize=(16, 8))
fig = sns.boxplot(x=var, y='SalePrice', data=data)
fig.axis(ymin=0, ymax=800000)
plt.xticks(rotation=90)
plt.savefig('../../reports/figures/house_prices_yearbuilt_box_plot.png')
plt.show()

#%% multivariate analysis
# correlation matrix
corrmat = df.corr()
f, ax = plt.subplots(figsize=(12, 9))
sns.heatmap(corrmat, vmax=.8, square=True)
plt.savefig('../../reports/figures/house_prices_correlation_matrix.png')
plt.show()

#%% saleprice correlation matrix
k = 10  # number of variables for heatmap
cols = corrmat.nlargest(k, 'SalePrice')['SalePrice'].index
cm = np.corrcoef(df[cols].values.T)
sns.set(font_scale=1.25)
hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10},
                 yticklabels=cols.values, xticklabels=cols.values)
plt.savefig('../../reports/figures/house_prices_saleprice_correlation_matrix.png')
plt.show()

#%% scatter plot between saleprice and correlated variables
sns.set()
cols = ['SalePrice', 'OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'FullBath', 'YearBuilt']
sns.pairplot(df[cols], size=2.5)
plt.savefig('../../reports/figures/house_prices_saleprice_scatter_plots.png')
plt.show()

#%% missing data
total = df.isnull().sum().sort_values(ascending=False)
percent = (df.isnull().sum()/df.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
print(missing_data.head(20))

#%% dealing with missing data
df = df.drop((missing_data[missing_data['Total'] > 1]).index, 1)
df = df.drop(df.loc[df['Electrical'].isnull()].index)
print(df.isnull().sum().max())

#%% outliers
# univariate analysis
saleprice_scaled = StandardScaler().fit_transform(df['SalePrice'][:, np.newaxis])
low_range = saleprice_scaled[saleprice_scaled[:, 0].argsort()][:10]
high_range = saleprice_scaled[saleprice_scaled[:, 0].argsort()][-10:]
print('Outer range (low) of the distribution:')
print(low_range)
print('\nOuter range (high) of the distribution:')
print(high_range)

#%% bivariate analysis
# saleprice/grlivarea
# deleting points
print(df.sort_values(by='GrLivArea', ascending=False)[:2])
df = df.drop(df[df['Id'] == 1299].index)
df = df.drop(df[df['Id'] == 524].index)

var = 'GrLivArea'
data = pd.concat([df['SalePrice'], df[var]], axis=1)
data.plot.scatter(x=var, y='SalePrice', ylim=(0, 800000))
plt.savefig('../../reports/figures/house_prices_saleprice_grlivarea_scatter_plot.png')
plt.show()

#%% normality
# saleprice
# histogram and normal probability plot
sns.distplot(df['SalePrice'], fit=norm)
plt.savefig('../../reports/figures/house_prices_saleprice_normality_histogram.png')
plt.show()

fig = plt.figure()
res = stats.probplot(df['SalePrice'], plot=plt)
plt.savefig('../../reports/figures/house_prices_saleprice_normal_probability_plot.png')
plt.show()

#%% log transformations
df['SalePrice'] = np.log(df['SalePrice'])

# transformed histogram and normal probability plot
sns.distplot(df['SalePrice'], fit=norm)
plt.savefig('../../reports/figures/house_prices_transformed_saleprice_normality_histogram.png')
plt.show()

fig = plt.figure()
res = stats.probplot(df['SalePrice'], plot=plt)
plt.savefig('../../reports/figures/house_prices_transformed_saleprice_normal_probability_plot.png')
plt.show()

#%% grlivarea
# histogram and normal probability plot
sns.distplot(df['GrLivArea'], fit=norm)
plt.savefig('../../reports/figures/house_prices_grlivarea_normality_histogram.png')
plt.show()

fig = plt.figure()
res = stats.probplot(df['GrLivArea'], plot=plt)
plt.savefig('../../reports/figures/house_prices_grlivarea_normal_probability_plot.png')
plt.show()

# log transformations
df['GrLivArea'] = np.log(df['GrLivArea'])

# transformed histogram and normal probability plot
sns.distplot(df['GrLivArea'], fit=norm)
plt.savefig('../../reports/figures/house_prices_transformed_grlivarea_normality_histogram.png')
plt.show()

fig = plt.figure()
res = stats.probplot(df['GrLivArea'], plot=plt)
plt.savefig('../../reports/figures/house_prices_transformed_grlivarea_normal_probability_plot.png')
plt.show()

#%% totalbsmtsf
# histogram and normal probability plot
sns.distplot(df['TotalBsmtSF'], fit=norm)
plt.savefig('../../reports/figures/house_prices_totalbsmtsf_normality_histogram.png')
plt.show()

fig = plt.figure()
res = stats.probplot(df['TotalBsmtSF'], plot=plt)
plt.savefig('../../reports/figures/house_prices_totalbsmtsf_normal_probability_plot.png')
plt.show()

# create new column
# if area > 0  it gets 1, for area == 0 it gets 0
df['HasBsmt'] = pd.Series(len(df['TotalBsmtSF']), index=df.index)
df['HasBsmt'] = 0
df.loc[df['TotalBsmtSF'] > 0, 'HasBsmt'] = 1

# log transformations
df.loc[df['HasBsmt'] == 1, 'TotalBsmtSF'] = np.log(df['TotalBsmtSF'])

# transformed histogram and normal probability plot
sns.distplot(df[df['TotalBsmtSF'] > 0]['TotalBsmtSF'], fit=norm)
plt.savefig('../../reports/figures/house_prices_transformed_totalbsmtsf_normality_histogram.png')
plt.show()

fig = plt.figure()
res = stats.probplot(df[df['TotalBsmtSF'] > 0]['TotalBsmtSF'], plot=plt)
plt.savefig('../../reports/figures/house_prices_transformed_totalbsmtsf_normal_probability_plot.png')
plt.show()

#%% homoscedasticity
# scatter plot
plt.scatter(df['GrLivArea'], df['SalePrice'])
plt.savefig('../../reports/figures/house_prices_transformed_saleprice_grlivarea_scatter_plot.png')
plt.show()

plt.scatter(df[df['TotalBsmtSF'] > 0]['TotalBsmtSF'], df[df['TotalBsmtSF'] > 0]['SalePrice'])
plt.savefig('../../reports/figures/house_prices_transformed_saleprice_totalbsmtsf_scatter_plot.png')
plt.show()

#%% convert categorical variable into dummy
df = pd.get_dummies(df)
print(df.head())
df.to_csv('../../data/processed/house_prices.csv', index=False)
