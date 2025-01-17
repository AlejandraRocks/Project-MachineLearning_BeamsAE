import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors

## read excel
script_dir = os.path.dirname(os.path.abspath(__file__))
excelpath = os.path.join(script_dir, 'BP - V33_EA_ch1.xlsx')
BP_ch1 = pd.read_excel(excelpath)
data = BP_ch1


# Filter rows where 'load_level' contains 'RISE'
data_rise = data[data['load level'].str.contains('RISE', case=False, na=False)]

# Select only relevant columns for clustering (excluding CH and THR)
columns_to_exclude = ['CH', 'THR']
data_rise_filtered = data_rise.drop(columns=columns_to_exclude, errors='ignore').loc[:, 'RISE':'ABS-ENERGY']
data_rise_filtered = data_rise_filtered.select_dtypes(include=[np.number])

# Standardize the data
scaler = StandardScaler()
scaled_data = scaler.fit_transform(data_rise_filtered)

#
# Function to perform clustering by load level and cycle
def perform_clustering(data, eps=2):
    clustering_results = {}

    for load_level in data['load level'].unique():
        subset = data[data['load level'] == load_level].copy()

        # Display the number of rows in the current level
        print(f"{load_level}: {len(subset)} rows")
        
        # Select only relevant columns for clustering
        relevant_columns = subset.drop(columns=columns_to_exclude, errors='ignore').loc[:, 'RISE':'ABS-ENERGY']
        numeric_data = relevant_columns.select_dtypes(include=[np.number])

        if numeric_data.empty:
            print(f"No numeric data found for {load_level}")
            continue

        # Standardize the data
        scaled_data = scaler.fit_transform(numeric_data)

        # Define a dynamic value for min_samples based on the level size
        min_samples = max(2, min(10, len(subset)))

        # Apply DBSCAN
        db = DBSCAN(eps=eps, min_samples=min_samples).fit(scaled_data)
        
        # Add clustering results
        subset.loc[:, 'cluster'] = db.labels_
        clustering_results[load_level] = subset

        # Display the number of clusters
        n_clusters = len(set(db.labels_)) - (1 if -1 in db.labels_ else 0)
        print(f"{load_level}: {n_clusters} cluster(s) (min_samples={min_samples})")

        # Calculate the percentage of each cluster
        cluster_counts = subset['cluster'].value_counts(normalize=True) * 100
        #print(f"Percentage of points per cluster in {load_level}:")
        #print(cluster_counts)

    return clustering_results

# Run clustering
clustering_results = perform_clustering(data_rise)

# Save the results
for load_level, df in clustering_results.items():
    output_file = f"clustering_results_{load_level}.xlsx"
    #df.to_excel(output_file, index=False)
    print(f"Results for {load_level} saved in {output_file}")
