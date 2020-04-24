import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MaxAbsScaler
from sklearn.cluster import KMeans

#%% load dataset
vehicles = pd.read_csv('../../data/raw/vehicles.csv')
print(vehicles.head())
print(vehicles.columns)
print(vehicles.shape)

#%% choosing select columns
select_columns = ['make', 'model', 'year', 'displ', 'cylinders', 'trany', 'drive', 'VClass', 'fuelType', 'barrels08',
                  'city08', 'highway08', 'comb08', 'co2TailpipeGpm', 'fuelCost08']
vehicles = vehicles[select_columns][vehicles.year <= 2016].drop_duplicates().dropna()
vehicles = vehicles.sort_values(['make', 'model', 'year'])

#%% rename the columns
vehicles.columns = ['Make', 'Model', 'Year', 'Engine Displacement', 'Cylinders', 'Transmission', 'Drivetrain',
                    'Vehicle Class', 'Fuel Type', 'Fuel Barrels/Year', 'City MPG', 'Highway MPG', 'Combined MPG',
                    'CO2 Emission Grams/Mile', 'Fuel Cost/Year']


#%% aggregating to higher level categories
def unique_col_values(df):
	for column in df:
		print(f'{df[column].name} | {len(df[column].unique())} | {df[column].dtype}')


unique_col_values(vehicles)

#%% aggregating transmission types
AUTOMATIC = 'Automatic'
MANUAL = 'Manual'

vehicles.loc[vehicles['Transmission'].str.startswith('A'), 'Transmission Type'] = AUTOMATIC
vehicles.loc[vehicles['Transmission'].str.startswith('M'), 'Transmission Type'] = MANUAL

print(vehicles['Transmission Type'].sample(5))

#%% aggregating vehicle class
small = ['Compact Cars', 'Subcompact Cars', 'Two Seaters', 'Minicompact Cars']
midsize = ['Midsize Cars']
large = ['Large Cars']

vehicles.loc[vehicles['Vehicle Class'].isin(small), 'Vehicle Category'] = 'Small Cars'
vehicles.loc[vehicles['Vehicle Class'].isin(midsize), 'Vehicle Category'] = 'Midsize Cars'
vehicles.loc[vehicles['Vehicle Class'].isin(large), 'Vehicle Category'] = 'Large Cars'
vehicles.loc[vehicles['Vehicle Class'].str.contains('Station'), 'Vehicle Category'] = 'Station Wagons'
vehicles.loc[vehicles['Vehicle Class'].str.contains('Truck'), 'Vehicle Category'] = 'Pickup Trucks'
vehicles.loc[vehicles['Vehicle Class'].str.contains('Special Purpose'), 'Vehicle Category'] = 'Special Purpose'
vehicles.loc[vehicles['Vehicle Class'].str.contains('Sport Utility'), 'Vehicle Category'] = 'Sport Utility'
vehicles.loc[vehicles['Vehicle Class'].str.lower().str.contains('van'), 'Vehicle Category'] = 'Vans & Minivans'

print(vehicles['Vehicle Category'].sample(5))

#%% aggregating make and model
vehicles['Model Type'] = (vehicles['Make'] + ' ' + vehicles['Model'].str.split().str.get(0))
print(vehicles['Model Type'].sample(5))

#%% aggregating fuel type
print(vehicles['Fuel Type'].unique())
vehicles['Gas'] = 0
vehicles['Ethanol'] = 0
vehicles['Electric'] = 0
vehicles['Propane'] = 0
vehicles['Natural Gas'] = 0

vehicles.loc[vehicles['Fuel Type'].str.contains('Regular|Gasoline|Midgrade|Premium|Diesel'), 'Gas'] = 1
vehicles.loc[vehicles['Fuel Type'].str.contains('E85'), 'Ethanol'] = 1
vehicles.loc[vehicles['Fuel Type'].str.contains('Electricity'), 'Electric'] = 1
vehicles.loc[vehicles['Fuel Type'].str.contains('propane'), 'Propane'] = 1
vehicles.loc[vehicles['Fuel Type'].str.contains('natural|CNG'), 'Natural Gas'] = 1
vehicles.loc[vehicles['Fuel Type'].str.contains('Regular|Gasoline'), 'Gas Type'] = 'Regular'
vehicles.loc[vehicles['Fuel Type'].str.contains('Midgrade'), 'Gas Type'] = 'Midgrade'
vehicles.loc[vehicles['Fuel Type'].str.contains('Premium'), 'Gas Type'] = 'Premium'
vehicles.loc[vehicles['Fuel Type'].str.contains('Diesel'), 'Gas Type'] = 'Diesel'
vehicles.loc[vehicles['Fuel Type'].str.contains('natural|CNG'), 'Gas Type'] = 'Natural'

cols = ['Fuel Type', 'Gas Type', 'Gas', 'Ethanol', 'Electric', 'Propane', 'Natural Gas']
print(vehicles[cols].sample(5))

#%% creating categories from continuous variables
# fuel efficiency
efficiency_categories = ['Very Low Efficiency', 'Low Efficiency', 'Moderate Efficiency', 'High Efficiency',
                         'Very High Efficiency']
vehicles['Fuel Efficiency'] = pd.qcut(vehicles['Combined MPG'], 5, efficiency_categories)
print(vehicles['Fuel Efficiency'].sample(5))

#%% engine size
engine_categories = ['Very Small Engine', 'Small Engine', 'Moderate Engine', 'Large Engine', 'Very Large Engine']
vehicles['Engine Size'] = pd.qcut(vehicles['Engine Displacement'], 5, engine_categories)
print(vehicles['Engine Size'].sample(5))

#%% emissions
emission_categories = ['Very Low Emissions', 'Low Emissions', 'Moderate Emissions', 'High Emissions',
                       'Very High Emissions']
vehicles['Emissions'] = pd.qcut(vehicles['CO2 Emission Grams/Mile'], 5, emission_categories)
print(vehicles['Emissions'].sample(5))

#%% fuel cost
fuelcost_categories = ['Very Low Fuel Cost', 'Low Fuel Cost', 'Moderate Fuel Cost', 'High Fuel Cost',
                       'Very High Fuel Cost']
vehicles['Fuel Cost'] = pd.qcut(vehicles['Fuel Cost/Year'], 5, fuelcost_categories)
print(vehicles['Fuel Cost'].sample(5))

#%%  clustering to create additional categories
cluster_columns = ['Engine Displacement', 'Cylinders', 'Fuel Barrels/Year', 'City MPG', 'Highway MPG', 'Combined MPG',
                   'CO2 Emission Grams/Mile', 'Fuel Cost/Year']

# scale the features
scaler = MaxAbsScaler()
vehicle_clusters = scaler.fit_transform(vehicles[cluster_columns])
vehicle_clusters = pd.DataFrame(vehicle_clusters, columns=cluster_columns)
print(vehicle_clusters.head())


def kmeans_cluster(df, n_clusters=2):
	model = KMeans(n_clusters=n_clusters, random_state=1)
	clusters = model.fit_predict(df)
	cluster_results = df.copy()
	cluster_results['Cluster'] = clusters
	return cluster_results


def summarize_clustering(results):
	cluster_size = results.groupby(['Cluster']).size().reset_index()
	cluster_size.columns = ['Cluster', 'Count']
	cluster_means = results.groupby(['Cluster'], as_index=False).mean()
	cluster_summary = pd.merge(cluster_size, cluster_means, on='Cluster')
	return cluster_summary


cluster_results = kmeans_cluster(vehicle_clusters, 4)
cluster_summary = summarize_clustering(cluster_results)
print(cluster_results)
print(cluster_summary)

#%% clusters visualization
sns.heatmap(cluster_summary[cluster_columns].transpose(), annot=True)
plt.savefig('../../reports/figures/vehicles_clusters_heatmap.png')
plt.show()

#%% assign descriptive names to clusters
cluster_results['Cluster Name'] = ''
cluster_results['Cluster Name'][cluster_results['Cluster'] == 0] = 'Midsized Balanced'
cluster_results['Cluster Name'][cluster_results['Cluster'] == 1] = 'Large Moderately Efficient'
cluster_results['Cluster Name'][cluster_results['Cluster'] == 2] = 'Small Very Efficient'
cluster_results['Cluster Name'][cluster_results['Cluster'] == 3] = 'large Inefficient'

vehicles = vehicles.reset_index().drop('index', axis=1)
vehicles['Cluster Name'] = cluster_results['Cluster Name']
print(vehicles['Cluster Name'].sample(5))


#%% making the data smaller
# aggregate count for vehicle categories in 2016
def agg_count(df, group_field):
	grouped = df.groupby(group_field, as_index=False).size()
	grouped = grouped.sort_values(ascending=False)
	grouped = pd.DataFrame(grouped).reset_index()
	grouped.columns = [group_field, 'Count']
	return grouped


vehicles_2016 = vehicles[vehicles['Year'] == 2016]
category_counts = agg_count(vehicles_2016, 'Vehicle Category')
print(category_counts)

ax = sns.barplot(data=category_counts, x='Count', y='Vehicle Category')
ax.set(xlabel='\n Number of Vehicles Manufactured')
plt.title('Vehicles Manufactured by Category (2016) \n')
plt.savefig('../../reports/figures/vehicles_2016_categories_barplot.png')
plt.show()

#%% aggregate count for vehicle categories in 1985
vehicles_1985 = vehicles[vehicles['Year'] == 1985]
category_counts = agg_count(vehicles_1985, 'Vehicle Category')
print(category_counts)

ax = sns.barplot(data=category_counts, x='Count', y='Vehicle Category')
ax.set(xlabel='\n Number of Vehicles Manufactured')
plt.title('Vehicles Manufactured by Category (1985) \n')
plt.savefig('../../reports/figures/vehicles_1985_categories_barplot.png')
plt.show()

#%% aggregate count for vehicle classes 2016
class_counts = agg_count(vehicles_2016, 'Vehicle Class')
print(class_counts)

ax = sns.barplot(data=class_counts, x='Count', y='Vehicle Class')
ax.set(xlabel='\n Number of Vehicles Manufactured')
plt.title('Vehicles Manufactured by Class (2016) \n')
plt.savefig('../../reports/figures/vehicles_class_barplot.png')
plt.show()

#%% aggregate count for vehicle manufacturers 2016
make_counts = agg_count(vehicles_2016, 'Make')
print(make_counts)

ax = sns.barplot(data=make_counts, x='Count', y='Make')
ax.set(xlabel='\n Number of Vehicles Manufactured')
plt.title('Vehicles Manufactured by Make (2016) \n')
plt.savefig('../../reports/figures/vehicles_make_barplot.png')
plt.show()

#%% aggregate count for vehicle manufacturers for high efficiency cars
very_efficient = vehicles[vehicles['Fuel Efficiency'] == 'Very High Efficiency']
make_counts = agg_count(very_efficient, 'Make')
print(make_counts)

ax = sns.barplot(data=make_counts, x='Count', y='Make')
ax.set(xlabel='\n Number of Vehicles Manufactured')
plt.title('Very Fuel Efficient vehicles by Make \n')
plt.savefig('../../reports/figures/vehicles_high_efficiency_make_barplot.png')
plt.show()


#%% getting aggregate average
def agg_avg(df, group_field, calc_field):
	grouped = df.groupby(group_field, as_index=False)[calc_field].mean()
	grouped = grouped.sort_values(calc_field, ascending=False)
	grouped.columns = [group_field, 'Avg ' + str(calc_field)]
	return grouped


category_avg_mpg = agg_avg(vehicles_2016, 'Vehicle Category', 'Combined MPG')

ax = sns.barplot(data=category_avg_mpg, x='Avg Combined MPG', y='Vehicle Category')
ax.set(xlabel='\n Average Combined MPG')
plt.title('Average Combined MPG by Category (2016) \n')
plt.savefig('../../reports/figures/vehicles_category_avg_mpg_barplot.png')
plt.show()


#%% pivoting the data for more detail
def pivot_count(df, rows, columns, calc_field):
	df_pivot = df.pivot_table(values=calc_field, index=rows, columns=columns, aggfunc=np.size).dropna(axis=0, how='all')
	return df_pivot


# pivot map for fuel efficiency against engine size in 2016
effic_size_pivot = pivot_count(vehicles_2016, 'Fuel Efficiency', 'Engine Size', 'Combined MPG')
print(effic_size_pivot)

sns.heatmap(effic_size_pivot, annot=True, fmt='g')
ax.set(xlabel='\n Engine Size')
plt.title('Fuel Efficiency vs. Engine Size (2016) \n')
plt.savefig('../../reports/figures/vehicles_fuel_efficiency_to_engine_2016_heatmap.png')
plt.show()

#%% pivot map for fuel efficiency against engine size in 1985
effic_size_pivot = pivot_count(vehicles_1985, 'Fuel Efficiency', 'Engine Size', 'Combined MPG')
print(effic_size_pivot)

fig, ax = plt.subplots(figsize=(15, 8))
sns.heatmap(effic_size_pivot, annot=True, fmt='g')
ax.set(xlabel='\n Engine Size')
plt.title('Fuel Efficiency vs. Engine Size (1985) \n')
plt.savefig('../../reports/figures/vehicles_fuel_efficiency_to_engine_1985_heatmap.png')
plt.show()

#%% pivot map for fuel efficiency and engine size against vehicle category in 2016
effic_size_category = pivot_count(vehicles_2016, ['Engine Size', 'Fuel Efficiency'], 'Vehicle Category', 'Combined MPG')
print(effic_size_category)

fig, ax = plt.subplots(figsize=(20, 10))
sns.heatmap(effic_size_category, annot=True, fmt='g')
ax.set(xlabel='\n Vehicle Category')
plt.title('Fuel Efficiency + Engine Size vs. Vehicle Category (2016) \n')
plt.savefig('../../reports/figures/vehicles_fuel_efficiency_and_engine_to_category_2016_heatmap.png')
plt.show()

#%% pivot map for fuel efficiency against engine size in 1985
effic_size_pivot = pivot_count(vehicles_2016, 'Make', 'Vehicle Category', 'Combined MPG')
print(effic_size_pivot)

fig, ax = plt.subplots(figsize=(20, 20))
sns.heatmap(effic_size_pivot, annot=True, fmt='g')
ax.set(xlabel='\n Vehicle Category')
plt.title('Make vs. Vehicle Category (2016) \n')
plt.savefig('../../reports/figures/vehicles_make_to_category_heatmap.png')
plt.show()


#%% visualizing changes over time
def multi_line(df, x, y):
	ax = df.groupby([x, y]).size().unstack(y).plot(figsize=(15, 8), cmap='Set2')


# line graph of vehicle categories against year
multi_line(vehicles, 'Year', 'Vehicle Category')
ax.set(xlabel='\n Year')
plt.title('Vehicle Categories Over Time \n')
plt.savefig('../../reports/figures/vehicles_categories_linegraph.png')
plt.show()

#%% line graph of BMW vehicle categories against year
bmw = vehicles[vehicles['Make'] == 'BMW']
multi_line(bmw, 'Year', 'Vehicle Category')
ax.set(xlabel='\n Year')
plt.title('BMW Vehicle Categories Over Time \n')
plt.savefig('../../reports/figures/vehicles_bmw_categories_linegraph.png')
plt.show()

#%% line graph of Toyota vehicle categories against year
toyota = vehicles[vehicles['Make'] == 'Toyota']
multi_line(toyota, 'Year', 'Vehicle Category')
ax.set(xlabel='\n Year')
plt.title('Toyota Vehicle Categories Over Time \n')
plt.savefig('../../reports/figures/vehicles_toyota_categories_linegraph.png')
plt.show()

#%% examining relationships between variables
select_columns = ['Engine Displacement', 'Cylinders', 'Fuel Barrels/Year', 'City MPG', 'Highway MPG', 'Combined MPG',
                  'CO2 Emission Grams/Mile', 'Fuel Cost/Year', 'Cluster Name']
sns.pairplot(vehicles[select_columns], size=3)
plt.savefig('../../reports/figures/vehicles_features_correlation.png')
plt.show()

#%% scatter plot of engine against mpg
sns.lmplot('Engine Displacement', 'Combined MPG', data=vehicles, hue='Cluster Name', size=8, fit_reg=False)
plt.savefig('../../reports/figures/vehicles_engine_scatter_plot.png')
plt.show()

#%% save dataset
print(vehicles.head())
print(vehicles.shape)
vehicles.to_csv('../../data/processed/vehicles.csv', index=False)
