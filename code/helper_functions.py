# define a tiny atomic function
def plot_top_n(well_data_df, top_k_index = pd.Int64Index([1]),  
               target_index = pd.Int64Index([1]), title = None):
    fig, ax = plt.subplots(figsize = (8,7))
    ax.scatter(well_data_df.loc[target_index,:].mean_ss_easting,
               well_data_df.loc[target_index,:].mean_ss_northing, 
               zorder=1, alpha= 1, c='b', s=50)
    ax.scatter(well_data_df.drop(target_index).loc[top_k_index,:].mean_ss_easting,
               well_data_df.drop(target_index).loc[top_k_index,:].mean_ss_northing, 
               zorder=1, alpha= 1, c='r', s=50)
    ax.scatter(well_data_df.drop(top_k_index).mean_ss_easting,
               well_data_df.drop(top_k_index).mean_ss_northing, 
               zorder=1, alpha= 0.5, c='k', s=10)
    ax.set_title(title)
    
# make a kNN predict function
def predict_print_knn(well_data_df, dist_func, k , target_index, column_str, title_str = ""):
    well_data_df = well_data_df.assign(distance = well_data_df.apply(
        dist_func, axis=1, v=well_data_df.loc[target_index,:]))
    
    top_k_index = well_data_df.nsmallest(k,'distance').index
    plot_top_n(well_data_df, top_k_index, target_index, title_str)
    
    # let's 'predict' the breakdown_isip_ratio - kinda a fluke
    pred = well_data_df.loc[top_k_index,column_str].mean()
    actual = well_data_df.loc[target_index,column_str]
    
    # print using f-strings
    print(title_str + ' kNN prediction: ' 
          + f'{pred:.3}' 
          + ' vs target: ' 
          + f'{actual:.3}')

# simple 2D colour plot
def plot_single_map(df, xyz_cols = ['x','y','first_order_residual']):
    fig, ax = plt.subplots(figsize = (8,7))
    ax.set_title(xyz_cols[2])
    ax.tripcolor(df[xyz_cols[0]], df[xyz_cols[1]], df[xyz_cols[2]])
    ax.set_aspect('equal')
    plt.tight_layout()
    plt.show()

# TODO: MAke this function to plot all plots...
def plot_four_maps(df):
    plt.figure(figsize=(10, 7))
    G = gridspec.GridSpec(2, 2)
    ax1 = plt.subplot(G[0, 0])
    ax2 = plt.subplot(G[0, 1])
    ax3 = plt.subplot(G[1, 0])
    ax4 = plt.subplot(G[1, 1])
    
    ax1.set_title(df.columns[3])
    ax1.tripcolor(df.iloc[:,1], df.iloc[:,2], df.iloc[:,3])
    ax1.set_aspect('equal')

    ax2.set_title(df.columns[4])
    ax2.tripcolor(df.iloc[:,1], df.iloc[:,2], df.iloc[:,4])
    ax2.set_aspect('equal')
    
    ax3.set_title(df.columns[5])
    ax3.tripcolor(df.iloc[:,1], df.iloc[:,2], df.iloc[:,5])
    ax3.set_aspect('equal')
    
    ax4.set_title(df.columns[6])
    ax4.tripcolor(df.iloc[:,1], df.iloc[:,2], df.iloc[:,6])
    ax4.set_aspect('equal')
    
    plt.tight_layout()
    plt.show()
    
def plot_pca_variance(pca):
    plt.figure(figsize=(10, 7))
    G = gridspec.GridSpec(1, 2)
    ax1 = plt.subplot(G[0, 0])
    ax2 = plt.subplot(G[0, 1])
    
    ax1.set_title('Individual Variance')
    ax1.set_xlabel('PCA')
    ax1.plot(pca.explained_variance_ratio_, 'k-')
    ax1.plot(pca.explained_variance_ratio_, 'ro')

    ax2.set_title('Cummulative Variance')
    ax2.plot(np.cumsum(pca.explained_variance_ratio_), 'k-')
    ax2.plot(np.cumsum(pca.explained_variance_ratio_), 'ro')
    
    plt.tight_layout()
    plt.show()
       
def plot_pca_unit_circle(pca, df):
    coefficients = np.transpose(pca.components_)
    
    ex_var_ratio = pca.explained_variance_ratio_
    
    pca_cols = ['PC-'+str(x) for x in range(len(ex_var_ratio))]
    
    pca_info = pd.DataFrame(coefficients, 
                            columns=pca_cols, 
                            index=df.columns)
    
    plt.Circle((0,0),radius=10, color='g', fill=False)
    circle1=plt.Circle((0,0),radius=1, color='g', fill=False)
    fig = plt.gcf()
    fig.gca().add_artist(circle1)
    for idx in range(len(pca_info["PC-0"])):
        x = pca_info["PC-0"][idx]
        y = pca_info["PC-1"][idx]
        plt.plot([0.0,x],[0.0,y],'k-')
        plt.plot(x, y, 'rx')
        plt.annotate(pca_info.index[idx], xy=(x,y))
    plt.xlabel("PC-0 (%s%%)" % str(ex_var_ratio[0])[:4].lstrip("0."))
    plt.ylabel("PC-1 (%s%%)" % str(ex_var_ratio[1])[:4].lstrip("0."))
    plt.xlim((-1,1))
    plt.ylim((-1,1))
    plt.axes.Axes.set_aspect('equal')
    plt.title("Circle of Correlations")
    
def plot_cluster_results(clust_res, df):
    plt.figure(figsize=(10, 7))
    G = gridspec.GridSpec(1, 2)
    ax1 = plt.subplot(G[0, 0])
    ax2 = plt.subplot(G[0, 1])
    
    clust_centres = pd.DataFrame(clust_res.cluster_centers_, 
                                 columns = df.columns)
    
    ax1.set_title('XY Scatter Plot')
    ax1.set_xlabel('PCA')
    ax1.scatter(df.loc[:,'mean_ss_easting'], 
                df.loc[:, 'mean_ss_northing'], c=clust_res.labels_)
    ax1.scatter(clust_centres.loc[:,'mean_ss_easting'], 
                clust_centres.loc[:, 'mean_ss_tvd'], c='red', s=300)
    
    ax2.set_title('XZ Scatter Plot')
    ax2.set_xlabel('PCA')
    ax2.scatter(df.loc[:,'mean_ss_easting'], 
                df.loc[:, 'mean_ss_tvd'], c=clust_res.labels_)
    ax2.scatter(clust_centres.loc[:,'mean_ss_easting'], 
                clust_centres.loc[:, 'mean_ss_tvd'], c='red', s=300)
    
    plt.tight_layout()
    plt.show()
    
def kmeans_elbowplot(df, kmax = 12):
    wcss = []
    silhouette = []
    calinski_harabaz = []
    for i in range(2, kmax):
        kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
        kmeans.fit(df)
        wcss.append(kmeans.inertia_)
        ch_score = metrics.calinski_harabaz_score(df, kmeans.labels_)
        sil_score = metrics.silhouette_score(df, kmeans.labels_, metric='euclidean')
        silhouette.append(sil_score)
        calinski_harabaz.append(ch_score)
    
    plt.figure(figsize=(10, 7))
    G = gridspec.GridSpec(1, 3)
    ax1 = plt.subplot(G[0, 0])
    ax2 = plt.subplot(G[0, 1])
    ax3 = plt.subplot(G[0, 2])
    
    ax1.set_title('WCSS')
    ax1.set_xlabel('PCA')
    ax1.plot(range(2, kmax), wcss, c='k')
    
    ax2.set_title('Silhouette Score')
    ax2.set_xlabel('PCA')
    ax2.plot(range(2, kmax), silhouette, c='r')

    ax3.set_title('Calinski Harabaz Score')
    ax3.set_xlabel('PCA')
    ax3.plot(range(2, kmax), calinski_harabaz, c='g')
    
    plt.tight_layout()
    plt.show()
    
def plot_dendrogram(df, method = 'ward'):
    linkage_matrix = linkage(df, method)
    figure = plt.figure(figsize=(7.5, 5))
    dendrogram(
        linkage_matrix,
        color_threshold=0,
    )
    plt.title('Hierarchical Clustering Dendrogram (' + method + ')')
    plt.xlabel('sample index')
    plt.ylabel('distance')
    plt.tight_layout()
    plt.show()