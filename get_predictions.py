from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input, decode_predictions
import numpy as np
from numpy import asarray
import os
from PIL import Image


def get_report_relative(model, img_name, n_clusters, class_of_interest, name_of_interest, mode):

    img_name = img_name.split('.')[0]

    img_path = 'images/predictions_'+str(n_clusters)+'/'+img_name+'/'
    text_file = open(img_path+"results_"+mode+".txt", "w")
    
    original_features = []
    
    image_original = []
    original_pix = []
    deltas = {}
    
    for file in os.listdir(img_path):
        if not file.endswith(mode+'.png') and not file.endswith(img_name+'.png') and not file.endswith('.jpg'):
            continue
        
        img = image.load_img(img_path+file, target_size=(224, 224))
        #plt.imshow(img)
        #plt.show()
        
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        features = model.predict(x)
        
        text_file.write(file+'\n')
        print("P: " + file)
    
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
    
    
    clustered_img = Image.open('images/clust_'+str(n_clusters)+'/'+img_name+'_'+str(n_clusters)+'.png') 
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
                        border = False;
                        if(i > 0 and clustered_pix[i-1,j] != c_tuple): border = True
                        #elif (i > 1 and clustered_pix[i-2,j] != c_tuple): border = True
                        elif (j > 0 and clustered_pix[i,j-1] != c_tuple): border = True
                        #elif (j > 1 and clustered_pix[i,j-2] != c_tuple): border = True
                        if(i < 223 and clustered_pix[i+1,j] != c_tuple): border = True
                        #elif (i < 222 and clustered_pix[i+2,j] != c_tuple): border = True
                        elif (j < 223 and clustered_pix[i,j+1] != c_tuple): border = True
                        #elif (j < 222 and clustered_pix[i,j+2] != c_tuple): border = True
                        threshold_1 = max(deltas.values())/10
                        threshold_2 = max(deltas.values())/1.2
                        if(border):
                            if(delta > threshold_1):                                
                                original_pix[i,j] = tuple((0, 500, 0))
                            elif(delta < -threshold_1):
                                original_pix[i,j] = tuple((500, 0, 0))
                            else: original_pix[i,j] = tuple((500, 500, 0))
                        else:
                            if(delta > threshold_2):                      
                                original_pix[i,j] = tuple(np.asarray(original_pix[i,j]) + (-100, 100, -100))
                            elif(delta > threshold_1):  
                                if(i%2 != 0 or j%2 != 0): original_pix[i,j] = tuple(np.asarray(original_pix[i,j]) + (0, 100, 0))                           
                            elif(delta < -threshold_2):
                                original_pix[i,j] = tuple(np.asarray(original_pix[i,j]) + (100, -100, -100)) 
                            elif(delta < -threshold_1):
                                if(i%2 != 0 or j%2 != 0): original_pix[i,j] = tuple(np.asarray(original_pix[i,j]) + (100, 0, 0))                              
                            else: original_pix[i,j] = tuple(np.asarray(original_pix[i,j]) + (100, 100, 0))  
    
    if not os.path.exists(img_path+'predizioni/'):
        os.makedirs(img_path+'predizioni/')
        
    image_original.save(img_path+'predizioni/'+img_name+'_'+name_of_interest+'_'+mode+'_relative.png')
        
def get_report_absolute(model, img_name, n_clusters, class_of_interest, name_of_interest, mode):

    img_name = img_name.split('.')[0]

    img_path = 'images/predictions_'+str(n_clusters)+'/'+img_name+'/'
    text_file = open(img_path+"results_"+mode+".txt", "w")
    
    original_features = []
    
    image_original = []
    original_pix = []
    deltas = {}
    
    for file in os.listdir(img_path):
        if not file.endswith(mode+'.png') and not file.endswith(img_name+'.png') and not file.endswith('.jpg'):
            continue
        
        img = image.load_img(img_path+file, target_size=(224, 224))
        #plt.imshow(img)
        #plt.show()
        
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        features = model.predict(x)
        
        text_file.write(file+'\n')
        print("P: " + file)
    
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
    
    
    clustered_img = Image.open('images/clust_'+str(n_clusters)+'/'+img_name+'_'+str(n_clusters)+'.png') 
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
                        border = False;
                        if(i > 0 and clustered_pix[i-1,j] != c_tuple): border = True
                        #elif (i > 1 and clustered_pix[i-2,j] != c_tuple): border = True
                        elif (j > 0 and clustered_pix[i,j-1] != c_tuple): border = True
                        #elif (j > 1 and clustered_pix[i,j-2] != c_tuple): border = True
                        if(i < 223 and clustered_pix[i+1,j] != c_tuple): border = True
                        #elif (i < 222 and clustered_pix[i+2,j] != c_tuple): border = True
                        elif (j < 223 and clustered_pix[i,j+1] != c_tuple): border = True
                        #elif (j < 222 and clustered_pix[i,j+2] != c_tuple): border = True
                        threshold_1 = 0.05
                        threshold_2 = .30
                        if(border):
                            if(delta > threshold_1):                                
                                original_pix[i,j] = tuple((0, 500, 0))
                            elif(delta < -threshold_1):
                                original_pix[i,j] = tuple((500, 0, 0))
                            else: original_pix[i,j] = tuple((500, 500, 0))
                        else:
                            if(delta > threshold_2):                      
                                original_pix[i,j] = tuple(np.asarray(original_pix[i,j]) + (-100, 100, -100))
                            elif(delta > threshold_1):  
                                if(i%2 != 0 or j%2 != 0): original_pix[i,j] = tuple(np.asarray(original_pix[i,j]) + (0, 100, 0))                           
                            elif(delta < -threshold_2):
                                original_pix[i,j] = tuple(np.asarray(original_pix[i,j]) + (100, -100, -100)) 
                            elif(delta < -threshold_1):
                                if(i%2 != 0 or j%2 != 0): original_pix[i,j] = tuple(np.asarray(original_pix[i,j]) + (100, 0, 0))                              
                            else: original_pix[i,j] = tuple(np.asarray(original_pix[i,j]) + (100, 100, 0))  
    
    if not os.path.exists(img_path+'predizioni/'):
        os.makedirs(img_path+'predizioni/')
        
    image_original.save(img_path+'predizioni/'+img_name+'_'+name_of_interest+'_'+mode+'_absolute.png')
        