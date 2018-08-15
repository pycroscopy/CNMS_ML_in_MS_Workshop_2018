# -*- coding: utf-8 -*-
"""
Utility functions for CNN-CAM tutorial

Created on Mon Jul 30 20:57:13 2018

@author: Maxim Ziatdinov
"""

import h5py
import numpy as np
import cv2

def resize_images(image_data, target_size):
    '''Resizes training images based on the the target size defined by user'''
    
    img_new = np.zeros((image_data.shape[0], target_size[0], target_size[1]))
    for idx, img in enumerate(image_data):
        img = cv2.resize(img, target_size, cv2.INTER_AREA)
        img_new[idx,:,:] = img
    return img_new


def load_training_data(filepath, target_size=(64,64)):
    '''Loads training data and corresponding labels'''
    
    defects_d = {}
    train_imgs = np.empty((0, target_size[0], target_size[1]))
    train_lbls = np.empty((0, 1))
    with h5py.File(filepath, 'r') as f:
        for im in f:
            defects_d[f[im+'/label_data'][0]] = im
            images = f[im+'/image_data'][:]
            if images.shape[1:3] != target_size:
                images = resize_images(images, target_size)
            train_imgs = np.append(train_imgs, images, axis = 0)
            labels = f[im+'/label_data'][:].reshape(-1,1)
            train_lbls = np.append(train_lbls, labels)
    print("Training data loaded.", "\nno. of images in the training set:",
          str(train_imgs.shape[0]), "\nresolution of each image:",
          str((train_imgs.shape[1], train_imgs.shape[2])))
    return train_imgs, train_lbls, defects_d
       

def tf_format(image_data, image_size):
    '''Change image format to keras/tensorflow format'''
    
    image_data = image_data.reshape(image_data.shape[0], image_size[0], image_size[1], 1)
    image_data = image_data.astype('float32')
    image_data = (image_data - np.amin(image_data))/np.ptp(image_data)
    return image_data