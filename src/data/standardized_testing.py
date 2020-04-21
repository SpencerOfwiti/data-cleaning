import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

#%% load data
sat_17 = pd.read_csv('../../data/raw/sat_2017.csv')
sat_18 = pd.read_csv('../../data/raw/sat_2018.csv')
act_17 = pd.read_csv('../../data/raw/act_2017.csv')
act_18 = pd.read_csv('../../data/raw/act_2018.csv')

print('SAT 2017 shape =', sat_17.shape)
print('SAT 2018 shape =', sat_18.shape)
print('ACT 2017 shape =', act_17.shape)
print('ACT 2018 shape =', act_18.shape)

#%% remove duplicate states
print(act_18['State'].value_counts())
print(act_18[act_18['State'] == 'Maine'])
act_18.drop(act_18.index[52], inplace=True)
act_18 = act_18.reset_index(drop=True)
print(act_18.shape)


#%% method for comparing columns in two dataframes
def compare_values(act_col, sat_col):
	act_vals = []
	sat_vals = []

	for a_val in act_col:
		act_vals.append(a_val)

	for s_val in sat_col:
		sat_vals.append(s_val)

	# display unique values in ACT column
	print('Values in ACT only: ')
	for val_a in act_vals:
		if val_a not in sat_vals:
			print(val_a)

	print('---------------------------------------')

	# display unique vals in SAT column
	print('Values in SAT only: ')
	for val_s in sat_vals:
		if val_s not in act_vals:
			print(val_s)


#%% comparing ACT and SAT state columns
compare_values(act_17['State'], sat_17['State'])
compare_values(act_18['State'], sat_18['State'])

#%% removing 'National' value in state column in both ACT dataframes
print(act_17[act_17['State'] == 'National'])
act_17.drop(act_17.index[0], inplace=True)
act_17 = act_17.reset_index(drop=True)
print(act_17.shape)

print(act_18[act_18['State'] == 'National'])
act_18.drop(act_18.index[23], inplace=True)
act_18 = act_18.reset_index(drop=True)
print(act_18.shape)

#%% removing inconsistencies in act_18 and sat_18
print(act_17[act_17['State'] == 'Washington, D.C.'])
print(act_17[act_17['State'] == 'District of Columbia'])
act_18['State'].replace({'Washington, D.C.': 'District of Columbia'}, inplace=True)

#%% addressing inconsistent number of columns
sat_17.drop(columns=['Evidence-Based Reading and Writing', 'Math'], inplace=True)
act_17.drop(columns=['English', 'Math', 'Reading', 'Science'], inplace=True)
sat_18.drop(columns=['Evidence-Based Reading and Writing', 'Math'], inplace=True)

print('SAT 2017 column names =', sat_17.columns, '\n')
print('SAT 2018 column names =', sat_18.columns, '\n')
print('ACT 2017 column names =', act_17.columns, '\n')
print('ACT 2018 column names =', act_18.columns, '\n')

#%% check for missing data
print('SAT 2017 Missing Data: ')
print(sat_17.isnull().sum(), '\n')
print('SAT 2018 Missing Data: ')
print(sat_18.isnull().sum(), '\n')
print('ACT 2017 Missing Data: ')
print(act_17.isnull().sum(), '\n')
print('ACT 2018 Missing Data: ')
print(act_18.isnull().sum(), '\n')

#%% check column datatypes
print('SAT 2017 Data Types: ')
print(sat_17.dtypes, '\n')
print('SAT 2019 Data Types: ')
print(sat_18.dtypes, '\n')
print('ACT 2017 Data Types: ')
print(act_17.dtypes, '\n')
print('ACT 2018 Data Types: ')
print(act_18.dtypes, '\n')


#%% method to remove percentage sign from participation column
def fix_participation(column):
	return column.apply(lambda cells: cells.strip('%'))


#%% method to convert columns to float
def convert_to_float(exam_df):
	features = [col for col in exam_df.columns if col != 'State']
	exam_df[features] = exam_df[features].astype(float)
	return exam_df


#%% fix participation
sat_17['Participation'] = fix_participation(sat_17['Participation'])
sat_18['Participation'] = fix_participation(sat_18['Participation'])
act_17['Participation'] = fix_participation(act_17['Participation'])
act_18['Participation'] = fix_participation(act_18['Participation'])

#%% remove anomalies in act_17
print(act_17['Composite'].value_counts())
act_17['Composite'] = act_17['Composite'].apply(lambda x_cell: x_cell.strip('x'))

#%% convert all columns except State to float
act_17 = convert_to_float(act_17)
sat_17 = convert_to_float(sat_17)
act_18 = convert_to_float(act_18)
sat_18 = convert_to_float(sat_18)

#%% rename the columns
new_act_17_cols = {
	'State': 'state',
	'Participation': 'act_participation_17',
	'Composite': 'act_composite_17'
}
act_17.rename(columns=new_act_17_cols, inplace=True)

new_sat_17_cols = {
	'State': 'state',
	'Participation': 'sat_participation_17',
	'Composite': 'sat_composite_17'
}
sat_17.rename(columns=new_sat_17_cols, inplace=True)

new_act_18_cols = {
	'State': 'state',
	'Participation': 'act_participation_18',
	'Composite': 'act_composite_18'
}
act_18.rename(columns=new_act_18_cols, inplace=True)

new_sat_18_cols = {
	'State': 'state',
	'Participation': 'sat_participation_18',
	'Composite': 'sat_composite_18'
}
sat_18.rename(columns=new_sat_18_cols, inplace=True)

#%% sort data by state column to prepare for merge
sat_17.sort_values(by=['state'], inplace=True)
sat_18.sort_values(by=['state'], inplace=True)
act_17.sort_values(by=['state'], inplace=True)
act_18.sort_values(by=['state'], inplace=True)

# reset index
sat_17 = sat_17.reset_index(drop=True)
sat_18 = sat_18.reset_index(drop=True)
act_17 = act_17.reset_index(drop=True)
act_18 = act_18.reset_index(drop=True)

#%% merge SAT and ACT 2017 data frames
sat_act_17 = pd.merge(sat_17, act_17, left_index=True, on='state', how='outer')
print(sat_act_17.head())

# merge SAT and ACT 2018 data frames
sat_act_18 = pd.merge(sat_18, act_18, left_index=True, on='state', how='outer')
print(sat_act_18.head())

df = pd.merge(sat_act_17, sat_act_18, left_index=True, on='state', how='outer')
print(df.shape)

#%% save data frame
print(df.head())
df.to_csv('../../data/processed/sat_act_2017_2018.csv', encoding='utf-8', index=False)
