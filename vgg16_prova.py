from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input, decode_predictions
import numpy as np
from load_vgg16 import load_vgg16

model = load_vgg16()

img_path = 'images/Immagine.png'

img = image.load_img(img_path, target_size=(224, 224))

x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)
features = model.predict(x)
    
print('Predizione: '+str(decode_predictions(features, 3))+'\n')