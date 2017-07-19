from PIL import Image
import os, random, numpy as np

path = 'images\\'
mode = 'transparent' #noise, blur, transparent, pixelate, inverse
#inc_exc = 'exc'

for file in os.listdir(path):
    if os.path.isdir(path+file):
        # skip directories
        continue
    name = file.split('.')[0]   
    path_img = 'images\\'+name+'.jpg'
    path_img_seg = 'images\\clust_10\\'+name+'_5_10.png'
    path_img_out = 'images\\predictions_10\\'
    
    im = Image.open(path_img)
    pix = im.load()
    
    im_seg = Image.open(path_img_seg)
    im_seg = im_seg.convert('RGBA')
    pix_seg = im_seg.load()
    colors = im_seg.getcolors()
    
    if not os.path.exists(path_img_out+name+'\\'+mode+'\\'):
        os.makedirs(path_img_out+name+'\\'+mode+'\\')
    
    for color in colors:
        color = color[1]
        for i in range(0,224):
            for j in range(0,224):
                if pix_seg[i,j] != color: #inc/exc
                    pix_seg[i,j] = pix[i,j]
                else:
                    if mode == 'noise':
                        noise_index = 8
                        try:
                            pix_seg[i,j] = pix[i+random.randint(-noise_index, noise_index),j+random.randint(-noise_index, noise_index)]
                        except:
                            pix_seg[i,j] = pix[i,j]
                    elif mode == 'blur':
                        blur_index = 8
                        my_color = np.asarray(pix[i,j])
                        n = 1
                        for k in range(-blur_index, blur_index):
                            for h in range(-blur_index, blur_index):
                                try:
                                    my_color += np.asarray(pix[i+k,j+h])
                                except:
                                    my_color += np.asarray(pix[i,j])
                                n += 1
                        my_color = my_color/n
                        pix_seg[i,j] = tuple(my_color.astype(int))
                    elif mode == 'inverse':
                        pix_seg[i,j] = tuple((255, 255, 255)-np.asarray(pix[i,j]))
                    elif mode == 'transparent':
                        pix_seg[i,j] = (255, 255, 255, 0) 
                    elif mode == 'pixelate':
                        k = i - (i % 4)
                        h = j - (j % 4)
                        pix_seg[i,j] = pix[k,h]
                    else:    
                        pix_seg[i,j] = (255, 255, 255, 255)    
        print(path_img_out+name+'\\'+mode+'\\'+name+'_'+str(color)+'_'+mode+'.png')
        im_seg.save(path_img_out+name+'\\'+mode+'\\'+name+'_'+str(color)+'_'+mode+'.png')
        im_seg = Image.open(path_img_seg)
        im_seg = im_seg.convert('RGBA')
        pix_seg = im_seg.load()
        
    im.save(path_img_out+name+'\\'+mode+'\\0'+name+'.png')
        
        
        
        
    