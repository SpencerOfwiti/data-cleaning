import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
from nltk.metrics import edit_distance

plt.style.use('ggplot')
matplotlib.rcParams['figure.figsize'] = (12, 8)

#%% create a dateframe
df = pd.read_csv('../../data/raw/russian_housing.csv')

# shape and datatypes of data
print(df.shape)
print(df.dtypes)

# select numeric columns
df_numeric = df.select_dtypes(include=[np.number])
numeric_cols = df_numeric.columns.values
print(numeric_cols)

# select non-numeric columns
df_non_numeric = df.select_dtypes(exclude=[np.number])
non_numeric_cols = df_non_numeric.columns.values
print(non_numeric_cols)

#%% MISSING DATA
# missing data heatmap
cols = df.columns[:30]  # first 30 columns
colours = ['#000099', '#ffff00']  # specify colors - yellow is missing, blue is not missing
sns.heatmap(df[cols].isnull(), cmap=sns.color_palette(colours))
plt.savefig('../../reports/figures/russian_housing_missing_data_heatmap.png')
plt.show()

#%% missing data percentage list
# for larger datasets where visualization can take too long
for col in df.columns:
	pct_missing = np.mean(df[col].isnull())
	print(col, '- {}%'.format(round(pct_missing*100)))

#%% missing data histogram
# first create missing indicator for features with missing data
for col in df.columns:
	missing = df[col].isnull()
	num_missing = np.sum(missing)

	if num_missing > 0:
		print('Created missing indicator for:', col)
		df['{}_ismissing'.format(col)] = missing

# then based on the indicator, plot the histogram of missing values
ismissing_cols = [col for col in df.columns if 'ismissing' in col]
df['num_missing'] = df[ismissing_cols].sum(axis=1)

df['num_missing'].value_counts().reset_index().sort_values(by='index').plot.bar(x='index', y='num_missing')
plt.savefig('../../reports/figures/russian_housing_missing_data_histogram.png')
plt.show()

#%% Handling missing data
# drop rows with a lot of missing values
ind_missing = df[df['num_missing'] > 35].index
df_less_missing_rows = df.drop(ind_missing, axis=0)

#%% drop the feature with a lot of missing data
# hospital_beds_raion has a lot of missing data
cols_to_drop = ['hospital_beds_raion']
df_less_ho_beds_raion = df.drop(cols_to_drop, axis=1)

#%% impute the missing data
# replace missing values with the median
med = df['life_sq'].median()
print(med)
df['life_sq'] = df['life_sq'].fillna(med)

#%% impute the missing values and create the missing value indicator variables for each numeric column
for col in numeric_cols:
	missing = df[col].isnull()
	num_missing = np.sum(missing)

	if num_missing > 0:  # only do the imputation for the columns that have missing values.
		print('Imputing missing values for:', col)
		df['{}_ismissing'.format(col)] = missing
		med = df[col].median()
		df[col] = df[col].fillna(med)

#%% impute the missing values and create the missing value indicator variables for each non-numeric column
for col in non_numeric_cols:
	missing = df[col].isnull()
	num_missing = np.sum(missing)

	if num_missing > 0:  # only do the imputation for the columns that have missing values.
		print('Imputing missing values for:', col)
		df['{}_ismissing'.format(col)] = missing

		top = df[col].describe()['top']  # impute with the most frequent value.
		df[col] = df[col].fillna(top)

#%% replace the missing data
# categorical
df['sub_area'] = df['sub_area'].fillna('_MISSING_')

# numeric
df['life_sq'] = df['life_sq'].fillna(-999)

#%% IRREGULAR DATA (OUTLIERS)
# histogram / box plot
# histogram of life_sq
df['life_sq'].hist(bins=100)
plt.savefig('../../reports/figures/russian_housing_irregular_data_histogram.png')
plt.show()

#%% box plot of life_sq
df.boxplot(column=['life_sq'])
plt.savefig('../../reports/figures/russian_housing_irregular_data_boxplot.png')
plt.show()

#%% descriptive statistics
df['life_sq'].describe()

#%% bar chart - distribution of a categorical variable
df['ecology'].value_counts().plot.bar()
plt.savefig('../../reports/figures/russian_housing_irregular_data_bar_chart.png')
plt.show()

#%% UNNECESSARY DATA
# uninformative / repetitive
num_rows = len(df.index)
low_information_cols = []

for col in df.columns:
	cnts = df[col].value_counts(dropna=False)
	top_pct = (cnts/num_rows).iloc[0]

	if top_pct > 0.95:
		low_information_cols.append(col)
		print('{0}: {1:.5f}%'.format(col, top_pct*100))
		print(cnts)
		print()

# irrelevant data

#%% duplicates
# all features based
# we know that column 'id' is unique, but what if we drop it?
df_dedupped = df.drop('id', axis=1).drop_duplicates()

print(df.shape, df_dedupped.shape)

#%% key features based
key = ['timestamp', 'full_sq', 'life_sq', 'floor', 'build_year', 'num_room', 'price_doc']
df.fillna(-999).groupby(key)['id'].count().sort_values(ascending=False).head(20)

# drop duplicates based on a subset of variables
df_dedupped2 = df.drop_duplicates(subset=key)

print(df.shape, df_dedupped2.shape)

#%% INCONSISTENT DATA
# capitalization - inconsistent use of upper and lower case
df['sub_area'].value_counts(dropna=False)

# make everything lower case
df['sub_area_lower'] = df['sub_area'].str.lower()
df['sub_area_lower'].value_counts(dropna=False)

#%% formats
df['timestamp_dt'] = pd.to_datetime(df['timestamp'], format='%Y-%m-%d')
df['year'] = df['timestamp_dt'].dt.year
df['month'] = df['timestamp_dt'].dt.month
df['weekday'] = df['timestamp_dt'].dt.weekday

print(df['year'].value_counts(dropna=False))
print()
print(df['month'].value_counts(dropna=False))

#%% categorical values
df_city_ex = pd.DataFrame(data={'city': ['torontoo', 'toronto', 'tronto', 'vancouver', 'vancover', 'vancouvr', 'montreal', 'calgary']})

df_city_ex['city_distance_toronto'] = df_city_ex['city'].map(lambda x: edit_distance(x, 'toronto'))
df_city_ex['city_distance_vancouver'] = df_city_ex['city'].map(lambda x: edit_distance(x, 'vancouver'))

msk = df_city_ex['city_distance_toronto'] <= 2
df_city_ex.loc[msk, 'city'] = 'toronto'

msk = df_city_ex['city_distance_vancouver'] <= 2
df_city_ex.loc[msk, 'city'] = 'vancouver'
print(df_city_ex)

# addresses
