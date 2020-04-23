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
plt.savefig('../../reports/figures/bigmart_histograms.png')
plt.show()

#%% density plots
df.plot(kind='density', subplots=True, layout=(3, 3), sharex=False)
plt.savefig('../../reports/figures/bigmart_density_plots.png')
plt.show()

#%% box plot
df.plot(kind='box', subplots=True, layout=(3, 3), sharex=False, sharey=False)
plt.savefig('../../reports/figures/bigmart_box_plots.png')
plt.show()

#%% bar graph
df['Item_Type'].value_counts().plot(kind='bar')
df['Outlet_Identifier'].value_counts().plot(kind='bar')
plt.savefig('../../reports/figures/bigmart_outlets_bar_graph.png')
plt.show()

#%% multivariate analysis
# categorical data
# bar plot
sns.barplot(x='Outlet_Identifier', y='Item_Outlet_Sales', data=df)
plt.savefig('../../reports/figures/bigmart_outlet sales.png')
plt.show()
sns.barplot(x='Outlet_Location_Type', y='Item_Outlet_Sales', data=df)
plt.savefig('../../reports/figures/bigmart_outlet_location sales.png')
plt.show()
sns.barplot(x='Outlet_Size', y='Item_Outlet_Sales', data=df)
plt.savefig('../../reports/figures/bigmart_outlet_size_sales.png')
plt.show()

#%% numeric data
# scatter plot
plt.scatter(df['Item_MRP'], df['Item_Outlet_Sales'])
plt.savefig('../../reports/figures/bigmart_item_mrp_scatter_plot.png')
plt.show()

#%% correlation
corr = df.corr()
sns.heatmap(corr, vmax=1., square=False)
plt.savefig('../../reports/figures/bigmart_correlation__map.png')
plt.show()
