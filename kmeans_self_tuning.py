from matplotlib import pyplot as plt
from sklearn.manifold import TSNE
from sklearn import manifold
from sklearn import cluster
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from _overlapped import NULL
import scipy as sp, numpy as np

def s_t_kmeans(n_clusters, m, iterations, img_name):
    cluster_labels=[]
    for i in range(0, iterations): 
        #n_clusters += 1       
        print("Clustering in corso...")
        kmeans = cluster.KMeans(n_clusters=n_clusters, max_iter=300, n_jobs=1, precompute_distances=True)
        labels = kmeans.fit_predict(m)
        centers = kmeans.cluster_centers_
        print(centers)
        print("Clustering completo")
        #silhouette = metrics.silhouette_samples(m,labels,metric='cityblock')
        #print(silhouette)
        cluster_labels.append(labels)
        print(cluster_labels)
        imcluster = np.zeros((224,224))
        imcluster = imcluster.reshape((224*224,))
        imcluster = labels
        sp.misc.imsave('images\\clust_'+str(n_clusters)+'\\'+img_name+'_'+str(i)+'.png', imcluster.reshape(224, 224))
    return cluster_labels[0]