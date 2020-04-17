import numpy as np
import pandas as pd

#%% create a dataframe
olympics_df = pd.read_csv('../../data/raw/olympics.csv', header=1)

#%% rename columns in header
new_names = {'Unnamed: 0': 'Country',
             '? Summer': 'Summer Olympics',
             '01 !': 'Gold',
             '02 !': 'Silver',
             '03 !': 'Bronze',
             '? Winter': 'Winter Olympics',
             '01 !.1': 'Gold.1',
             '02 !.1': 'Silver.1',
             '03 !.1': 'Bronze.1',
             '? Games': '# Games',
             '01 !.2': 'Gold.2',
             '02 !.2': 'Silver.2',
             '03 !.2': 'Bronze.2'}
olympics_df.rename(columns=new_names, inplace=True)

#%% save dataframe
print(olympics_df.head())
olympics_df.to_csv('../../data/processed/olympics.csv', index=False)
