#1. Extract Columns
def get_cols(string, data):
    cols = [col for col in data.columns if string in col]
    return(cols)

#2. Normalise data
def normalise_data(data, col):
    
    data[col] = (data[col] - data[col].min()) / (data[col].max() - data[col].min())    
    
    return(data[col])
    
data[col_list] = normalise_data(data, col_list)

#Initiate variables
var_list = []
uniq_id= ''
num_clusters = 4
run_version = 3
datainclusion = ['full', 'bias']

kmeans_cluster_label = 'clus_kmeans' + "_" + datainclusion[1]+ "_" +str(num_clusters)+ "_r" +str(run_version)
agg_cluster_label = 'clus_hclust' + "_" + datainclusion[1]+ "_" +str(num_clusters)+ "_r" +str(run_version)

#Function to get clusters
def get_clusters(data, num_clusters, var_list):
    
    # hierarchical Clustering
    cluster = AgglomerativeClustering(n_clusters=num_clusters, affinity='euclidean', linkage='ward')  
    clustering = cluster.fit(data[var_list])

    #Kmeans Clustering
    kmeans = KMeans(n_clusters=num_clusters)
    kmeans.fit(data[var_list])

    data[kmeans_cluster_label] = kmeans.predict(data[var_list])

    data[agg_cluster_label] = cluster.fit_predict(data[var_list])
    
    return(data)
    
  #Function to print elbowcurve
  def get_elbowcurve(data, var_list):
    sse = {}
    for k in range(1, 12):
        kmeans = KMeans(n_clusters=k, max_iter=1000).fit(data[var_list])
        data["clusters"] = kmeans.labels_
        #print(data["clusters"])
        sse[k] = kmeans.inertia_ # Inertia: Sum of distances of samples to their closest cluster center

    plt.figure()
    plt.plot(list(sse.keys()), list(sse.values()))
    plt.xlabel("Number of cluster")
    plt.ylabel("SSE")
    plt.show()

#Visualisation of Clusters
def get_clusters_distplot(data_withclusters, var, label):
      
    # Sort the dataframe by target
    target_0 = data_withclusters.loc[data_withclusters[target] == 0]
    target_1 = data_withclusters.loc[data_withclusters[target] == 1]
    target_2 = data_withclusters.loc[data_withclusters[target] == 2]
    target_3 = data_withclusters.loc[data_withclusters[target] == 3]


    sns.distplot(target_0[[var]], hist=False, rug=False, color = 'red')
    sns.distplot(target_1[[var]], hist=False, rug=False, color = 'green')
    sns.distplot(target_2[[var]], hist=False, rug=False, color = 'blue')
    sns.distplot(target_3[[var]], hist=False, rug=False, color = 'yellow')
    
    plt.xlabel(var)
    plt.ylabel("Distribution")

###############################Execution###############################
# Get clusters
data_withclusters = get_clusters(data, 
                                 num_clusters=num_clusters, 
                                 var_list=var_list)

#Basic decriptives of clusters
data_withclusters.groupby([kmeans_cluster_label])[var_list].mean().reset_index()

#Print Elbow curve
get_elbowcurve(data, var_list)

#Distribution plot of Clusters 
for var in var_list:
    plt.figure()
    get_clusters_distplot(data_withclusters,
                          label= kmeans_cluster_label, 
                          var = var)
