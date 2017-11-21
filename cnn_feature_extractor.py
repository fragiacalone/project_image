from keras import backend as K
import cv2
import numpy as np  
import scipy as sp
import os.path
from kmeans_self_tuning import kmeans_clustering
from numpy.distutils.lib2def import output_def

def extract_features(model, image_name, n_clusters):
    image_path = 'images/input_images/'
    activation_path = 'images/total_activation/'
    output_path = 'images/clust_' + str(n_clusters) + '/'
    
    out_file_name = output_path + image_name.split('.')[0] + '_' + str(n_clusters) + '.png'
    
    if os.path.isfile(out_file_name):
        print(out_file_name + " already exists")
        return
    
    im_original = cv2.resize(cv2.imread(image_path+image_name), (224, 224))
    im = cv2.cvtColor(im_original, cv2.COLOR_BGR2RGB)
    im = im.transpose((2,1,0))
    im_converted = np.expand_dims(im, axis=0)
    
    layers_extract = [20, 29]
    hc = extract_hypercolumn(model, layers_extract, im_converted)
    #ave = np.average(hc.transpose(2, 1, 0), axis=2)
    #path = activation_path + image_name.split('.')[0] + '_'+str(layers_extract[0])+'_'+str(layers_extract[1])+'.png'
    #print("Activation img: "+path) 
    #sp.misc.imsave(path, ave.reshape(224, 224))
    m = hc.transpose(2,1,0).reshape(50176, -1)
    cluster_labels = kmeans_clustering(n_clusters, m)
    imcluster = np.zeros((224,224))
    imcluster = imcluster.reshape((224*224,))
    imcluster = cluster_labels
    print(out_file_name) 
    sp.misc.imsave(out_file_name, imcluster.reshape(224, 224))

def extract_hypercolumn(model, layer_indexes, instance):
        layers = [model.layers[li].output for li in layer_indexes]
        #get_feature = theano.function([model.layers[0].input], layers, allow_input_downcast=False)
        #feature_maps = get_feature(instance)
        get_feature = K.function([model.layers[0].input], layers)
        feature_maps = get_feature([instance,0])
        hypercolumns = []
        for convmap in feature_maps:      
            for fmap in convmap[0]:
                upscaled = sp.misc.imresize(fmap, size=(224, 224), mode="F", interp='bilinear')
                hypercolumns.append(upscaled)
        return np.asarray(hypercolumns)
