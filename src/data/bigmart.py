import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#%% create dataframe
df = pd.read_csv('../../data/raw/bigmart.csv')
print(df.head())
print(df.shape)
print(df.columns)

#%% Descriptive analysis
# numeric columns
print(df.describe())

#%% categorical variables
cat = df.dtypes[df.dtypes == 'object'].index
print(df[cat].describe())

#%% data exploration using visualization
# histogram of all numeric variables
df.hist()
plt.show()

#%% density plots
df.plot(kind='density', subplots=True, layout=(3, 3), sharex=False)
plt.show()

#%% box plot
df.plot(kind='box', subplots=True, layout=(3, 3), sharex=False, sharey=False)
plt.show()

#%% bar graph
df['Item_Type'].value_counts().plot(kind='bar')
df['Outlet_Identifier'].value_counts().plot(kind='bar')
plt.show()

#%% multivariate analysis
# categorical data
# bar plot
sns.barplot(x='Outlet_Identifier', y='Item_Outlet_Sales', data=df)
sns.barplot(x='Outlet_Location_Type', y='Item_Outlet_Sales', data=df)
sns.barplot(x='Outlet_Size', y='Item_Outlet_Sales', data=df)
plt.show()

#%% numeric data
# scatter plot
plt.scatter(df['Item_MRP'], df['Item_Outlet_Sales'])
plt.show()

#%% correlation
corr = df.corr()
sns.heatmap(corr, vmax=1., square=False)
plt.show()
