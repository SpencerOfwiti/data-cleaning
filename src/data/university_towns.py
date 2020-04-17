import pandas as pd
import numpy as np

#%% read data from txt file
university_towns = []
with open('../../data/raw/university_towns.txt') as file:
	for line in file:
		if '[edit]' in line:
			# remember this state until the next is found
			state = line
		else:
			# otherwise, we have a city; keep state as last-seen
			university_towns.append((state, line))

#%% create a dataframe
towns_df = pd.DataFrame(university_towns, columns=['State', 'RegionName'])


#%% function to get city and state only
def get_citystate(item):
	if ' (' in item:
		return item[:item.find(' (')]
	elif '[' in item:
		return item[:item.find('[')]
	else:
		return item


towns_df = towns_df.applymap(get_citystate)

#%% save dataframe
print(towns_df.head())
towns_df.to_csv('../../data/processed/university_towns.csv', index=False)
