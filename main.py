import os
from load_vgg16 import load_vgg16
from cnn_feature_extractor import extract_features
from get_features import perturbate_features, get_features
from get_predictions import get_report
from class_list import get_class


n_clusters = 10
perturbations = ['blur']#noise, blur, transparent, pixelate, inverse

model = load_vgg16()
for image_name in  os.listdir('images/input_images'):#, ['mouse.jpg']:#'space shuttle.jpg', 'pizza.jpg', 'mantis.jpg', 'lemon.jpg', 'gondola.jpg', 'corn.jpg', 'crosswords.jpg', 'bee.jpg', 'beer glass.jpg', 'sax.jpg', 'typewriter keyboard.jpg', 'starfish.jpg']: #  
    [class_of_interest, name_of_interest] = get_class(image_name.split('.')[0])    
    if(class_of_interest != -1):
        print("Class of interest: " + name_of_interest)       
        extract_features(model, image_name, n_clusters)
        for mode in perturbations:
            get_features(image_name, n_clusters)
            perturbate_features(image_name, mode, n_clusters)
            get_report(model, image_name, n_clusters, class_of_interest, name_of_interest, mode)
    #class_of_interest = 150
    #name_of_interest = get_name(class_of_interest)
        
    
    
