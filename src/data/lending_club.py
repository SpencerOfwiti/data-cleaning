import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

pd.set_option('max_columns', 120)
pd.set_option('max_colwidth', 5000)
plt.rcParams['figure.figsize'] = (12, 8)


#%% import dataset from data world
def initial_data_dump():
	data = pd.read_csv('https://query.data.world/s/lh4dob56x7uf2xrjuct2rq624dw5nf', skiprows=1, low_memory=False)
	dict = pd.read_csv('https://query.data.world/s/6jwx73kjh3etb33akvyxhfeuubpjmu')

	# save dataset and dictionary to raw data folder
	data.to_csv('../../data/raw/lending_club.csv', index=False)
	dict.to_csv('../../data/raw/lending_club_data_dictionary.csv', index=False)


# initial_data_dump()

#%% create dataframe
df = pd.read_csv('../../data/raw/lending_club.csv')
half_count = len(df) / 2
df = df.dropna(thresh=half_count, axis=1)  # drop any column with more than 50% missing data
df = df.drop(['url', 'desc'], axis=1)  # these columns are not useful for our purposes
print(df.describe())
print(df.shape)

#%% load data dictionary
data_dictionary = pd.read_csv('../../data/raw/lending_club_data_dictionary.csv')
data_dictionary = data_dictionary.rename(columns={'LoanStatNew': 'name', 'Description': 'description'})
print(data_dictionary.head())
print(data_dictionary.shape)
print(data_dictionary.columns.tolist())


#%% preview datatypes for columns
def col_mapping(df):
	df_dtypes = pd.DataFrame(df.dtypes, columns=['dtypes'])
	df_dtypes = df_dtypes.reset_index()
	df_dtypes['name'] = df_dtypes['index']
	df_dtypes = df_dtypes[['name', 'dtypes']]
	df_dtypes['first_value'] = df.loc[0].values
	preview = df_dtypes.merge(data_dictionary, on='name', how='left')
	return preview


preview = col_mapping(df)
print(preview.head())
print(preview.shape)

#%% dropping irrelevant columns
drop_list = ['id', 'member_id', 'funded_amnt', 'funded_amnt_inv', 'int_rate', 'sub_grade', 'emp_title', 'issue_d',
             'zip_code', 'out_prncp', 'out_prncp_inv', 'total_pymnt', 'total_pymnt_inv', 'total_rec_prncp',
             'total_rec_int', 'total_rec_late_fee', 'recoveries', 'collection_recovery_fee', 'last_pymnt_d',
             'last_pymnt_amnt']
df = df.drop(drop_list, axis=1)

#%% investigating FICO score columns
print(df['fico_range_low'].unique())
print(df['fico_range_high'].unique())

fico_columns = ['fico_range_high', 'fico_range_low']
print(df.shape[0])
df.dropna(subset=fico_columns, inplace=True)
print(df.shape[0])
df[fico_columns].plot.hist(alpha=0.5, bins=20)
plt.savefig('../../reports/figures/lending_club_fico.png')
plt.show()

df['fico_average'] = (df['fico_range_high'] + df['fico_range_low']) / 2
cols = ['fico_range_low', 'fico_range_high', 'fico_average']
print(df[cols].head())

# drop columns in favour of average
drop_cols = ['fico_range_low', 'fico_range_high', 'last_fico_range_low', 'last_fico_range_high']
df = df.drop(drop_cols, axis=1)

#%% preparing the target variable
print(preview[preview.name == 'loan_status'])
meaning = ["Loan has been fully paid off.",
           "Loan for which there is no longer a reasonable expectation of further payments.",
           "While the loan was paid off, the loan application today would no longer meet the credit policy and wouldn't be approved on to the marketplace.",
           "While the loan was charged off, the loan application today would no longer meet the credit policy and wouldn't be approved on to the marketplace.",
           "Loan is up to date on current payments.",
           "The loan is past due but still in the grace period of 15 days.",
           "Loan hasn't been paid in 31 to 120 days (late on the current payment).",
           "Loan hasn't been paid in 16 to 30 days (late on the current payment).",
           "Loan is defaulted on and no payment has been made for more than 121 days."]
status, count = df['loan_status'].value_counts().index, df['loan_status'].value_counts().values
loan_statuses_explanation = pd.DataFrame({'Loan Status': status, 'Count': count, 'Meaning': meaning})[['Loan Status', 'Count', 'Meaning']]
print(loan_statuses_explanation)

# selecting only rows with fully paid or charged off loans
df = df[(df['loan_status'] == 'Fully Paid') | (df['loan_status'] == 'Charged Off')]
mapping_dictionary = {'loan_status': {'Fully Paid': 1, 'Charged Off': 0}}
df = df.replace(mapping_dictionary)

#%% visualize target variable outcomes
fig, axs = plt.subplots(1, 2, figsize=(14, 7))
sns.countplot(x='loan_status', data=df, ax=axs[0])
axs[0].set_title('Frequency of each Loan Status')
df.loan_status.value_counts().plot(x=None, y=None, kind='pie', ax=axs[1], autopct='%1.2f%%')
axs[1].set_title('Percentage of each Loan Status')
plt.savefig('../../reports/figures/lending_club_loan_status.png')
plt.show()

#%% Remove columns with only one value
df = df.loc[:, df.apply(pd.Series.nunique) != 1]

# view columns with less than 4 unique values
for col in df.columns:
	if len(df[col].unique()) < 4:
		print(df[col].value_counts())
		print()

print(df.shape[1])
df = df.drop('pymnt_plan', axis=1)
print("We've been able to reduce the features to =>", df.shape[1])

#%% save dataframe display first rows and shape
print(df.head())
print(df.shape)
df.to_csv('../../data/interim/lending_club.csv', index=False)

#%% map dictionary to current features and save
dict = col_mapping(df)
print(dict.head())
dict.to_csv('../../data/interim/lending_club_data_dictionary.csv', index=False)

#%% preparing the features for machine learning
data = pd.read_csv('../../data/interim/lending_club.csv')
print(data.head())
print(data.shape)

#%% handle missing values
null_counts = data.isnull().sum()
print('Number of null values in each column:\n', null_counts)

# drop with at least 1% null
# percent = len(data) * 0.99
# data = data.dropna(thresh=percent, axis=1)
data = data.drop('pub_rec_bankruptcies', axis=1)

# drop all rows with a missing value
data = data.dropna()
print(data.shape)

#%% investigating categorical columns
print('Data types and their frequency\n', data.dtypes.value_counts())

# columns of datatype object
object_columns_df = data.select_dtypes(include=['object'])
print(object_columns_df.iloc[0])

# convert revol_util to float
data['revol_util'] = data['revol_util'].str.rstrip('%').astype('float')

#%% explore unique value counts of 6 columns with categorical values
cols = ['home_ownership', 'grade', 'verification_status', 'emp_length', 'term', 'addr_state']
for name in cols:
	print(name, ':')
	print(object_columns_df[name].value_counts(), '\n')

#%% get unique value counts between purpose and title columns
for name in ['purpose', 'title']:
	print('Unique values in column:', name, '\n')
	print(data[name].value_counts(), '\n')

drop_cols = ['last_credit_pull_d', 'addr_state', 'title', 'earliest_cr_line']
data = data.drop(drop_cols, axis=1)

#%% convert ordinal categorical columns to numeric features
mapping_dict = {
	"emp_length": {
		"10+ years": 10,
		"9 years": 9,
		"8 years": 8,
		"7 years": 7,
		"6 years": 6,
		"5 years": 5,
		"4 years": 4,
		"3 years": 3,
		"2 years": 2,
		"1 year": 1,
		"< 1 year": 0,
		"n/a": 0
	},
	"grade": {
		"A": 1,
		"B": 2,
		"C": 3,
		"D": 4,
		"E": 5,
		"F": 6,
		"G": 7
	}
}

# map columns to appropriate values
data = data.replace(mapping_dict)
print(data[['emp_length', 'grade']].head())

#%% map dictionary to current features and save
data_dict = col_mapping(data)
print(data_dict.head())
print(data_dict.shape)
data_dict.to_csv('../../data/processed/lending_club_data_dictionary.csv', index=False)

#%% convert nominal features into numeric values
nominal_columns = ["home_ownership", "verification_status", "purpose", "term"]
dummy_df = pd.get_dummies(data[nominal_columns])
data = pd.concat([data, dummy_df], axis=1)
data = data.drop(nominal_columns, axis=1)

#%% save cleaned data
print(data.head())
print(data.info())
print(data.shape)
data.to_csv('../../data/processed/lending_club.csv', index=False)
