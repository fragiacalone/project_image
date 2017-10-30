from PIL import Image
import os, random, numpy as np

def perturbate_features(image_name, mode, n_clusters):
    image_name = image_name.split('.')[0]
    path_img = 'images/input_images/'+image_name+'.jpg'
    path_img_seg = 'images/clust_'+str(n_clusters)+'/'+image_name+'_'+str(n_clusters)+'.png'
    path_img_out = 'images/predictions_'+str(n_clusters)+'/'
    seg_path = path_img_out+image_name+'/'
       
    im = Image.open(path_img)
    pix = im.load()
    
    im_seg = Image.open(path_img_seg)
    im_seg = im_seg.convert('RGBA')
    pix_seg = im_seg.load()
    colors = im_seg.getcolors()
    
    if not os.path.exists(seg_path):
        os.makedirs(seg_path)
    
    if os.path.isfile(seg_path+'/'+image_name+'_(0, 0, 0, 255)_'+mode+'.png'):
        print("Perturbations already done")       
    else:  
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
            print(path_img_out+image_name+'/'+image_name+'/'+image_name+'_'+str(color)+'_'+mode+'.png')
            
            if not os.path.exists(seg_path):
                os.makedirs(seg_path)
            im_seg.save(seg_path+image_name+'_'+str(color)+'_'+mode+'.png')
            im_seg = Image.open(path_img_seg)
            im_seg = im_seg.convert('RGBA')
            pix_seg = im_seg.load()
            
        im.save(seg_path+'/0riginal_'+image_name+'.png')
        
        
        
        
    