import pandas as pd
import numpy as np

#%% create a dataframe
# making a list of missing value types
missing_values = ['n/a', 'na', '--']
df = pd.read_csv('../../data/raw/property_data.csv', na_values=missing_values)
print(df.head())

#%% standard missing values
# checking for ST_NUM column
print(df['ST_NUM'])
print(df['ST_NUM'].isnull())

#%% non-standard missing values
# checking for NUM_BEDROOMS column
print(df['NUM_BEDROOMS'])
print(df['NUM_BEDROOMS'].isnull())

# checking for SQ_FT column
print(df['SQ_FT'])
print(df['SQ_FT'].isnull())

#%% unexpected missing value
# detecting numbers
count = 0
for row in df['OWN_OCCUPIED']:
	try:
		int(row)
		df.loc[count, 'OWN_OCCUPIED'] = np.nan
	except ValueError:
		pass
	count += 1

# detecting words
count = 0
for row in df['NUM_BATH']:
	try:
		float(row)
	except ValueError:
		df.loc[count, 'NUM_BATH'] = np.nan
	count += 1

# checking for OWN_OCCUPIED column
print(df['OWN_OCCUPIED'])
print(df['OWN_OCCUPIED'].isnull())

# checking for NUM_BATH column
print(df['NUM_BATH'])
print(df['NUM_BATH'].isnull())

#%% summarising missing values
# total missing values for each feature
print(df.isnull().sum())

#%% any missing values?
print(df.isnull().values.any())

#%% total number of missing values
print(df.isnull().sum().sum())

#%% replacing values
# replace missing values with a number
df['ST_NUM'].fillna(125, inplace=True)

#%% location based replacement
df.loc[2, 'ST_NUM'] = 125

#%% replace using median
median = df['NUM_BEDROOMS'].median()
df['NUM_BEDROOMS'].fillna(median, inplace=True)

#%% replace using interpolation
interpolation = df['PID'].interpolate()
df['PID'].fillna(interpolation, inplace=True)

#%% replace with default value
default = 1
df['NUM_BATH'].fillna(default, inplace=True)

#%% save dataframe
print(df)
df.to_csv('../../data/processed/property_data.csv', index=False)
