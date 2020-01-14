# -*- coding: utf-8 -*-
"""
Wrapper for entire demo of untapped energy meetup
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import scipy.spatial.distance as ssd
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
# part 2 #
from sklearn.decomposition import PCA
import matplotlib.mlab as mlab
# part 3 #
from sklearn.cluster import KMeans
from sklearn import metrics
from sklearn.cluster import DBSCAN
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import linkage, dendrogram

exec(open('helper_functions.py').read())

### PART 1 ###

well_data = pd.read_csv('../data/bcogc_well_comp_info.csv')

# encode dates properly
date_cols = ['frac_start_date','frac_end_date', 'on_prod_date']

# a couple nice examples of method chaining in python
well_data.loc[:,date_cols] = (
        well_data
        .loc[:,date_cols]
        .apply(pd.to_datetime, errors='coerce')
        .apply(pd.to_numeric, errors='coerce')
        .div(31556952)
        )

# select numeric data, replace NaN and Inf values with zero
num_well_data = (well_data
                 .select_dtypes(include = [np.number])
                 .replace([np.inf, -np.inf], np.nan)
                 .fillna(0)
                 .copy())

#choose columns
select_col = ['mean_ss_tvd', 'mean_ss_easting', 'mean_ss_northing', 
             'on_prod_date', 'frac_start_date','calc_completed_length_m', 
             'mean_proppant_per_stage_t', 'calc_total_proppant_t',
             'total_gas_injected_m3', 'mean_fluid_per_stage_m3',
             'calc_total_fluid_m3', 'avg_stage_length', 'avg_stage_spacing',
             'mean_rate_m3_min', 'mean_stage_duration_min', 'mean_breakdown_mpa', 
             'mean_isip_mpa', 'fluid_intensity_comp_length_m3_m',
             'proppant_intensity_comp_length_t_m', 'frac_duration_days', 
             'breakdown_isip_ratio', 'min_midpoint_dist', 'horiz_wells_in_10km', 
             'first_order_residual', 'isotherm', 'paleozoic_structure', 
             'raw_montney_top', 'third_order_residual', 'n_quakes']

# make a select dataframe
select_well_data = num_well_data[num_well_data
                                 .columns
                                 .intersection(select_col)]

# pick the closest well to the centre of our area as the target
# use euclidean distance
coord_cols = ['mean_ss_tvd', 'mean_ss_easting', 'mean_ss_northing']
centroid = select_well_data.loc[:,coord_cols].mean()

target_index = (select_well_data
                .loc[:,coord_cols]
                .apply(ssd.euclidean, axis=1, v=centroid)
                .argmin())

# get distance
dist_func = ssd.euclidean

select_well_data = select_well_data.assign(distance = select_well_data.apply(
        dist_func, axis=1, v=select_well_data.loc[target_index,:]))

# distance on unscaled data
top_k_index = select_well_data.nsmallest(10,'distance').index
plot_top_n(select_well_data, top_k_index, target_index, 'Unscaled Euclidean')

# let's 'predict' the breakdown_isip_ratio - kinda a fluke
pred = select_well_data.loc[top_k_index,'breakdown_isip_ratio'].mean()
actual = select_well_data.loc[target_index,'breakdown_isip_ratio']

# print using f-strings
print('Unscaled kNN prediction: ' + f'{pred:.3}' + ' vs target: ' + f'{actual:.3}')

# let's scale the data using sklearn and try it again
scaled_well_data = pd.DataFrame(
        StandardScaler()
        .fit_transform(select_well_data), 
        columns=select_well_data.columns)

# fortunately we are using functions - works exactly the same as above
predict_print_knn(well_data_df = scaled_well_data, dist_func = ssd.euclidean,
                  k = 10, target_index = target_index, 
                  column_str = 'breakdown_isip_ratio', 
                  title_str = 'Scaled Euclidean')

# Use the sklearn methods
X = scaled_well_data.drop('breakdown_isip_ratio', axis=1)
y = scaled_well_data['breakdown_isip_ratio']
sk_knn = KNeighborsRegressor(n_neighbors=10).fit(X,y)
pred = sk_knn.predict(scaled_well_data
               .iloc[[target_index]]
               .drop('breakdown_isip_ratio',axis=1))
pred = np.asscalar(pred)
actual = scaled_well_data.loc[target_index,'breakdown_isip_ratio']
print('sklearn kNN prediction: ' + f'{pred:.3}' + ' vs target: ' + f'{actual:.3}')

# try a new distance function
predict_print_knn(well_data_df = scaled_well_data, dist_func = ssd.cosine,
                  k = 10, target_index = target_index, 
                  column_str = 'breakdown_isip_ratio', 
                  title_str = 'Scaled Cosine')

### PART 2 ### PCA

geo_maps = pd.read_csv('../data/raster_stack.csv')
plot_four_maps(geo_maps)

# Use sklearn to do some PCA
geomap_array = geo_maps.iloc[:,3:8]
pca = PCA(n_components=4).fit(geomap_array)
pca_maps = geo_maps.iloc[:,0:3].copy()

for i in range(0,4):
    pca_maps['pca_' + str(i+1)] = pca.transform(geomap_array)[:,i]

# show the four maps to explain how PCAs work, along with some additional 
# images
plot_four_maps(pca_maps)

# show an elbom plot with individual contributions and cumsum
plot_pca_variance(pca)

# plot a unit circle plot
plot_pca_unit_circle(pca, geomap_array)

# Lets do the entire unscaled and scaled dataset
pca_unscaled = PCA(n_components=10).fit(select_well_data.values)
plot_pca_variance(pca_unscaled)
plot_pca_unit_circle(pca_unscaled, select_well_data)

# This shows why scale is important
pca_scaled = PCA(n_components=10).fit(scaled_well_data.values)
plot_pca_variance(pca_scaled)
plot_pca_unit_circle(pca_scaled, scaled_well_data)

### PART 3 ### CLUSTERING
# we will use the scaled well completion dataset from here on
# say we want to cluster a section of wells into three type curves instead of one
# k-means
kmeans_2 = KMeans(n_clusters=2, random_state=0).fit(scaled_well_data)
plot_cluster_results(kmeans_2, scaled_well_data)

kmeans_3 = KMeans(n_clusters=3, random_state=0).fit(scaled_well_data)
plot_cluster_results(kmeans_3, scaled_well_data)

kmeans_5 = KMeans(n_clusters=5, random_state=0).fit(scaled_well_data)
plot_cluster_results(kmeans_5, scaled_well_data)

kmeans_3plus = KMeans(n_clusters=3, init='k-means++', max_iter=300, 
                n_init=10, random_state=0).fit(scaled_well_data)
plot_cluster_results(kmeans_3plus, scaled_well_data)

#within cluster sum of squared error (i.e. withinness)
kmeans.inertia_
kmeans_elbowplot(X)

#TODO : Put a good dbscan example in
db = DBSCAN(eps=20, min_samples=10).fit(X)
plot_cluster_results(db, scaled_well_data)

### PART 4 ### HIERARCHICAL CLUSTERING

# "ward" minimizes the variance of the clusters being merged.
ward_hclust = AgglomerativeClustering(n_clusters=3, linkage='ward').fit(X)
plot_dendrogram(X, method = 'ward')

# "average" uses the average of the distances of each observation of the two sets.
avg_hclust = AgglomerativeClustering(n_clusters=3, linkage='average').fit(X)
plot_dendrogram(X, method = 'average')

# "complete" or maximum linkage uses the maximum distances between all observations.
comp_hclust = AgglomerativeClustering(n_clusters=3, linkage='complete').fit(X)
plot_dendrogram(X, method = 'complete')


