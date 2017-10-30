import os
from load_vgg16 import load_vgg16
from cnn_feature_extractor import extract_features
from get_features import perturbate_features
from get_predictions import get_report_absolute, get_report_relative
from class_list import get_class

n_clusters = 10
perturbations = ['transparent', 'blur']#noise, blur, transparent, pixelate, inverse

model = load_vgg16()
for image_name in ['goldfish.jpg']: #os.listdir('images/input_images'):
    [class_of_interest, name_of_interest] = get_class(image_name.split('.')[0])    
    if(class_of_interest != -1):
        print("Class of interest: " + name_of_interest)       
        extract_features(model, image_name, n_clusters)
        for mode in perturbations:
            perturbate_features(image_name, mode, n_clusters)
            get_report_absolute(model, image_name, n_clusters, class_of_interest, name_of_interest, mode)
            get_report_relative(model, image_name, n_clusters, class_of_interest, name_of_interest, mode)
    
    #class_of_interest = 150
    #name_of_interest = get_name(class_of_interest)
        
    
    
