from matplotlib import pyplot as plt
from sklearn.manifold import TSNE
from sklearn import manifold
from sklearn import cluster
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from _overlapped import NULL
import scipy as sp, numpy as np, os

def s_t_kmeans(n_clusters, m):
    text_file = open("n_clusters.txt", "w") 
    cluster_labels=np.zeros((224, 224))
    
    path='images/clust_'+str(n_clusters)+'/'
    if not os.path.exists(path):
        os.makedirs(path)
    
    bestval = 0
    best_n = -1
    imcluster = []
    values = []
    
    for n in range(n_clusters[0], n_clusters[1]):   
        text_file.write('Numero di cluster: '+str(n)+'\n')
        print("Clustering in corso...")
        kmeans = cluster.KMeans(n_clusters=n, max_iter=300, n_jobs=1, precompute_distances=True)
        labels = kmeans.fit_predict(m)
        print("Clustering completo")
        silhouette = metrics.silhouette_samples(m,labels,metric='euclidean')
        w = np.zeros(11)
        for value in silhouette:
            if value <= 0:
                w[0] = w[0]+1;
            elif value < 0.1:
                w[1] = w[1]+1;
            elif value < 0.2:
                w[2] = w[2]+1;
            elif value < 0.3:
                w[3] = w[3]+1;
            elif value < 0.4:
                w[4] = w[4]+1;
            elif value < 0.5:
                w[5] = w[5]+1;
            elif value < 0.6:
                w[6] = w[6]+1;
            elif value < 0.7:
                w[7] = w[7]+1;
            elif value < 0.8:
                w[8] = w[8]+1;
            elif value < 0.9:
                w[9] = w[9]+1;
            else:
                w[10] = w[10]+1;
        text_file.write(str(w)+'\n')
        val = 0
        for i in range(1, 10):
            val += i * w[i] 
        val = val / 55
        values.append(val)
        text_file.write(str(val)+'\n\n')     
        print(str(n)+': '+str(val))
        print('')
        if val > bestval:
            bestval = val
            best_n = n
            cluster_labels = labels    

        imcluster.append(labels)
        n += 1
        
    imcluster = np.asarray(imcluster)    
    sp.misc.imsave(path+str(n_clusters[0])+'_'+str(n_clusters[1])+'_'+str(best_n)+'.png', imcluster.reshape(224*(n_clusters[1]-n_clusters[0]), 224))
    text_file.write('Best choice: ' + str(best_n) + ' clusters')
    print('Best choice: ' + str(best_n) + ' clusters')
    text_file.close()
    
    values = np.asarray(values)
    intervals = np.linspace(n_clusters[0], n_clusters[1]-1, n_clusters[1]-n_clusters[0])
    plt.plot(intervals, values, 'ro')
    plt.axis([n_clusters[0]-1, n_clusters[1], 10, 50])
    plt.savefig(path+'clusters.jpg')
    
    return cluster_labels