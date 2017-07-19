from matplotlib import pyplot as plt
import theano
from keras import backend as K
import cv2, os
import numpy as np  
import scipy as sp
from load_vgg16 import load_vgg16
from kmeans_self_tuning import s_t_kmeans

image_name = 'carbonara.jpg' #carbonara, elephant, guinea, guitar, hotdog, lamp, martello, me, mouse, pizza, sax, sunglasses
image_path = 'images\\'
activation_path = 'images\\total_activation\\'
output_path = 'images\\clust_10\\'
n_clusters = 10


im_original = cv2.resize(cv2.imread(image_path+image_name), (224, 224))
im = cv2.cvtColor(im_original, cv2.COLOR_BGR2RGB)

#plt.imshow(im)
#plt.show() 

im = im.transpose((2,1,0))
im_converted = np.expand_dims(im, axis=0)

model = load_vgg16()
out = model.predict(im_converted)
#plt.plot(out.ravel())
#plt.show()

'''
def get_activations(model,layer_idx, X_batch):
    get_activations = K.function([model.layers[0].input,], [model.layers[layer_idx].output])
    activations = get_activations([X_batch,0])
    return activations


#    get_feature = theano.function([model.layers[0].input], model.layers[3].get_output(train=False), allow_input_downcast=False)
#    feat = get_feature(im)
feat = get_activations(model, 3, im_converted)
plt.imshow(feat[0][2])
plt.show()

#    get_feature = theano.function([model.layers[0].input], model.layers[15].get_output(train=False), allow_input_downcast=False)
#    feat = get_feature(im)
feat = get_activations(model, 15, im_converted)
plt.imshow(feat[0][13])
plt.show()
'''

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

'''
layers_extract = [0, 9]
hc1 = extract_hypercolumn(model, layers_extract, im_converted)
ave = np.average(hc1.transpose(2, 1, 0), axis=2)
path = activation_path + image_name.split('.')[0] + '_0_9.png'
print(path) 
sp.misc.imsave(path, ave.reshape(224, 224))
#plt.imshow(ave)
#plt.show()

layers_extract = [8, 12]
hc2 = extract_hypercolumn(model, layers_extract, im_converted)
ave = np.average(hc2.transpose(2, 1, 0), axis=2)
path = activation_path + image_name.split(sep='.')[0] + '_8_12.png'
print(path) 
sp.misc.imsave(path, ave.reshape(224, 224))
#plt.imshow(ave)
#plt.show()


layers_extract = [10, 19]
hc3 = extract_hypercolumn(model, layers_extract, im_converted)
ave = np.average(hc3.transpose(2, 1, 0), axis=2)
path = activation_path + image_name.split('.')[0] + '_10_19.png'
print(path) 
sp.misc.imsave(path, ave.reshape(224, 224))
#plt.imshow(ave)
#plt.show()

layers_extract = [18, 22]
hc4 = extract_hypercolumn(model, layers_extract, im_converted)
ave = np.average(hc4.transpose(2, 1, 0), axis=2)
path = activation_path + image_name.split('.')[0] + '_18_22.png'
print(path) 
sp.misc.imsave(path, ave.reshape(224, 224))#plt.imshow(ave)
#plt.show()
'''

layers_extract = [20, 29]
hc5 = extract_hypercolumn(model, layers_extract, im_converted)
ave = np.average(hc5.transpose(2, 1, 0), axis=2)
path = activation_path + image_name.split('.')[0] + '_20_29.png'
print(path) 
sp.misc.imsave(path, ave.reshape(224, 224))
#plt.imshow(ave)
#plt.show()
'''
layers_extract = [0, 29]
hctot = extract_hypercolumn(model, layers_extract, im_converted)
ave = np.average(hctot.transpose(2, 1, 0), axis=2)
path = activation_path + image_name.split('.')[0] + '.png'
print(path) 
sp.misc.imsave(path, ave.reshape(224, 224))
'''
i = 1
for hc in [hc5]:#[hc1, hc2, hc3, hc4, hc5]:
    m = hc.transpose(2,1,0).reshape(50176, -1)
    cluster_labels = s_t_kmeans(n_clusters, m, 3, image_name)
    imcluster = np.zeros((224,224))
    imcluster = imcluster.reshape((224*224,))
    imcluster = cluster_labels
    path = output_path + image_name.split('.')[0] + '_' + str(i) + '_' + str(n_clusters) + '.png'
    print(path) 
    sp.misc.imsave(path, imcluster.reshape(224, 224))
    i = i + 1
    #plt.imshow(imcluster.reshape(224, 224), cmap="hot")
    #plt.show()

os.system("pause")
