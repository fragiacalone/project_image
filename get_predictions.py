from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input, decode_predictions
import numpy as np
from numpy import asarray
import os
from PIL import Image
from load_vgg16 import load_vgg16

model = load_vgg16()
mode = 'blur' #noise, blur, transparent, pixelate, inverse
img_name = 'sunglasses' #carbonara, elephant, guinea, guitar, hotdog, lamp, martello, me, mouse, pizza, sax, sunglasses

class_of_interest = 836 #'carbonara' num-1
name_of_interest = 'sunglass'

n_clusters = 10

img_path = 'images\\predictions_10\\'+img_name+'\\'+mode+'\\'
text_file = open(img_path+"results.txt", "w")

original_features = []

image_original = []
original_pix = []
deltas = {}

for file in os.listdir(img_path):
    if not file.endswith('.png') and not file.endswith('.jpg'):
        continue
    
    img = image.load_img(img_path+file, target_size=(224, 224))
    #plt.imshow(img)
    #plt.show()
    
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    features = model.predict(x)
    
    text_file.write(file+'\n')
    print(file)

    text_file.write('Predizione: '+str(decode_predictions(features, 3))+'\n')
    
    if file.startswith('0'):
        original_features = features
        image_original = Image.open(img_path+file)
        original_pix = image_original.load()
    else:
        delta = original_features - features
        deltas[file.split('_')[1]] = delta[0][class_of_interest]
        text_file.write('Delta: '+str(decode_predictions(delta, 3))+'\n')
    text_file.write('\n')

text_file.close()



clustered_img = Image.open('images\\clust_'+str(n_clusters)+'\\'+img_name+'_5_'+str(n_clusters)+'.png') 
clustered_img = clustered_img.convert('RGBA')
clustered_pix = clustered_img.load()

for color in deltas:  
    c_tuple = color.replace('(', '').replace(')', '').replace(',', '')
    c_tuple = c_tuple.split(' ')
    c_tuple = np.asarray(c_tuple)
    c_tuple = tuple(c_tuple.astype(int))

    for i in range(0,224):
            for j in range(0,224):
                if clustered_pix[i,j] == c_tuple:
                    delta = deltas.get(color)
                    if(delta > 0):
                        original_pix[i,j] = tuple(np.asarray(original_pix[i,j]) + (0, 100+200*int(delta/100), 0))
                    else:
                        original_pix[i,j] = tuple(np.asarray(original_pix[i,j]) + (100+200*int(delta/100), 0, 0))

if not os.path.exists(img_path+'predizioni\\'):
    os.makedirs(img_path+'predizioni\\')
    
image_original.save(img_path+'predizioni\\'+img_name+'_'+name_of_interest+'.png')
    