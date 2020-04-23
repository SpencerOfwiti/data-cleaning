import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

#%% load dataset
digits = pd.read_csv('../../data/raw/digits.csv', header=None)
print(digits.head())
print(digits.describe())

#%% sampling the data
print(digits.sample(5))

#%% check for missing values
print(digits.isnull().sum())

#%% dimension reduction
# Principal Component Analysis (PCA)
# build the model
pca = PCA(n_components=2)

# reduce the data, output is ndarray
reduced_data = pca.fit_transform(digits)

# inspect shape of reduced data
print(reduced_data.shape)
print(reduced_data)

#%% visualize the data
labels = digits[64]
plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c=labels, cmap='viridis')
plt.savefig('../../reports/figures/digits_scatter_plot.png')
plt.show()
