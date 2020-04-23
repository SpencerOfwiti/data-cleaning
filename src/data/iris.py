import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
from bokeh.plotting import figure, output_file, show

#%% read dataset
header = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'class']
iris = pd.read_csv('../../data/raw/iris.csv', names=header)
print(iris.head())
print(iris.describe())

#%% performing queries
# petal length greater than sepal length
print(iris.query('petal_length > sepal_length'))

# petal length equals sepal length
print(iris.query('petal_length == sepal_length'))

#%% check for missing values
print(iris.isnull().sum())

#%% feature engineering
# factorizing class column
labels, levels = pd.factorize(iris['class'])
iris['class'] = labels
print(iris['class'].sample(5))

#%% feature selection
# using random forest algorithm
# isolate data, class and column values
x = iris.iloc[:, 0:4]
y = iris.iloc[:, -1]
names = iris.columns.values

# build the model
rfc = RandomForestClassifier()

# fit the model
rfc.fit(x, y)

# print the results
print('Features sorted by their score:')
print(sorted(zip(map(lambda x: round(x, 4), rfc.feature_importances_), names), reverse=True))

#%% visualize feature selection
# isolate feature importances
importance = rfc.feature_importances_

# sort the feature importances
sorted_importances = np.argsort(importance)

# insert padding
padding = np.arange(len(names)-1) + 0.5

# plot the data
plt.barh(padding, importance[sorted_importances], align='center')
plt.yticks(padding, names[sorted_importances])
plt.xlabel('Relative Importance')
plt.title('Variable Importance')
plt.savefig('../../reports/figures/iris_variable_importance.png')
plt.show()

#%% correlation identification with bokeh
# construct the scatter plot
p = figure(plot_width=700, plot_height=450, title='Petal Length vs Petal Width', tools='hover')
p.scatter(x=iris['petal_length'], y=iris['petal_width'])
p.xaxis.axis_label = 'Sepal Length'
p.yaxis.axis_label = 'Sepal Width'

# output the file
output_file('../../reports/iris_scatter.html')
show(p)

#%% correlation identification with pandas
# Pearson correlation
print('Pearson Correlation:')
print(iris.corr())

# rank data
iris = iris.rank()

# Kendall Tau correlation
print('Kendall Tau Correlation:')
print(iris.corr('kendall'))

# Spearman Rank correlation
print('Spearman Correlation:')
print(iris.corr('spearman'))
