from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input, decode_predictions
import numpy as np
from numpy import asarray
import os, csv
from PIL import Image


def get_report(model, img_name, n_clusters, class_of_interest, name_of_interest, mode):

    img_name = img_name.split('.')[0]

    img_path = 'images/predictions_'+str(n_clusters)+'/'+img_name+'/'
    text_file = open(img_path+'/'+img_name+'_'+name_of_interest+'_'+mode+'_ratio.txt', 'w', newline='')
    
    original_predictions = []
    
    image_original = []
    original_pix = []
    ratios = {}
    csvReport = open('images/reports/'+img_name+'_'+name_of_interest+'_'+mode+'_ratio.csv', 'w', newline='')
    csvReport.write(name_of_interest+";Prediction;Difference;Specific Ratio;Average Ratio;Weighted Avg Ratio;Caratterizzante\n")
    
    for file in os.listdir(img_path):
        ratio = []
        if not file.endswith(mode+'.png') and not file.endswith(img_name+'.png') and not file.endswith('.jpg'):
            continue
        
        img = image.load_img(img_path+file, target_size=(224, 224))
        #plt.imshow(img)
        #plt.show()
        ri = -1
        rim = -1
        riw = -1
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        predictions = model.predict(x)

        text_file.write(file+'\n')
        print("P: " + file)
    
        text_file.write('Predizione: '+str(decode_predictions(predictions, 3))+'\n')
        
        if file.startswith('0'):
            original_predictions = predictions
            image_original = Image.open(img_path+file)
            original_pix = image_original.load()
        else:
            ratio = original_predictions/predictions
            ri = ratio[0][class_of_interest]
            rim = np.average(ratio[0])
            riw = sum(ratio[0][g] * original_predictions[0][g] / sum(original_predictions[0]) for g in range(len(ratio[0])))
            ratios[file.split('_')[1]] = ri

        text_file.write('\n')
        p = predictions[0][class_of_interest] * 100
        d = original_predictions[0][class_of_interest]*100 - predictions[0][class_of_interest]*100
        caratt = ri/riw
        csvReport.write(file.split('_')[1] + ";" + str(p) + ";" + str(d) + ";" + str(ri) + ";" + str(rim)+";" + str(riw)+";"+str(caratt)+"\n")
    text_file.close()
    csvReport.close()
    
    clustered_img = Image.open('images/clust_'+str(n_clusters)+'/'+img_name+'_'+str(n_clusters)+'.png') 
    clustered_img = clustered_img.convert('RGBA')
    clustered_pix = clustered_img.load()
    
    for color in ratios:  
        c_tuple = color.replace('(', '').replace(')', '').replace(',', '')
        c_tuple = c_tuple.split(' ')
        c_tuple = np.asarray(c_tuple)
        c_tuple = tuple(c_tuple.astype(int))
    
        for i in range(0,224):
                for j in range(0,224):
                    if clustered_pix[i,j] == c_tuple:
                        r = ratios.get(color)
                        border = False;
                        if(i > 0 and clustered_pix[i-1,j] != c_tuple): border = True
                        elif (j > 0 and clustered_pix[i,j-1] != c_tuple): border = True
                        if(i < 223 and clustered_pix[i+1,j] != c_tuple): border = True
                        elif (j < 223 and clustered_pix[i,j+1] != c_tuple): border = True
                        
                        offset = int(np.average(np.asarray(original_pix[i,j])))
                        if(r > 1):   
                            addendum = int((1*r)*30)                          
                            original_pix[i,j] = (offset+30-addendum, offset+30+addendum, offset - 80)
                        else:    
                            addendum = int((1/r)*40)               
                            original_pix[i,j] = (offset+40+addendum, offset+40-addendum, offset - 80)
    
    if not os.path.exists(img_path+'predizioni/'):
        os.makedirs(img_path+'predizioni/')
        
    image_original.save(img_path+'predizioni/'+img_name+'_'+name_of_interest+'_'+mode+'_ratio.png')
    image_original.save('images/reports/'+img_name+'_'+name_of_interest+'_'+mode+'_ratio.png')