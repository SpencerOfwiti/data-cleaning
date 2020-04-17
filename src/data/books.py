import numpy as np
import pandas as pd

#%% dropping columns in a dataframe
df = pd.read_csv('../../data/raw/BL-Flickr-Images-Book.csv')
to_drop = ['Edition Statement',
           'Corporate Author',
           'Corporate Contributors',
           'Former owner',
           'Engraver',
           'Contributors',
           'Issuance type',
           'Shelfmarks']
df.drop(to_drop, inplace=True, axis=1)

#%% changing the index of a dataframe
# asserting if Identifier is unique
print(df['Identifier'].is_unique)  # True
df.set_index('Identifier', inplace=True)
print(df.loc[206])  # access via index
print(df.iloc[1])  # position-based indexing

#%% changing date of publication into numeric values
extr = df['Date of Publication'].str.extract(r'^(\d{4})', expand=False)
df['Date of Publication'] = pd.to_numeric(extr)
print(df['Date of Publication'].dtype)
print(df['Date of Publication'].isnull().sum() / len(df))

#%% cleaning place of publication
pub = df['Place of Publication']
london = pub.str.contains('London')
oxford = pub.str.contains('Oxford')
df['Place of Publication'] = np.where(london, 'London',
                                      np.where(oxford, 'Oxford',
                                               pub.str.replace('-', ' ')))

#%% save dataframe
print(df.head())
df.to_csv('../../data/processed/books.csv', index=False)
