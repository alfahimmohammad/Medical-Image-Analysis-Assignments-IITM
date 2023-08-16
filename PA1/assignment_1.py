# -*- coding: utf-8 -*-
"""
Created on Sun Sep  5 11:51:01 2021

@author: alfah
"""
#importing required libraries
#I have renamed the images as '1.png' and '2.png' for convenience
import cv2 
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
#%%
#Part 1
img = cv2.imread('60/1.png',0)
cx, cy = img.shape[0]/2, img.shape[1]/2
#creating affine matrices
a = np.array(([1,0,5.5],[0,1,4.4]), dtype='float')
b1 = np.array(([[np.cos(np.deg2rad(35)), np.sin(np.deg2rad(35)), (1 - np.cos(np.deg2rad(35)))*cx - np.sin(np.deg2rad(35))*cy],[-np.sin(np.deg2rad(35)), np.cos(np.deg2rad(35)), np.sin(np.deg2rad(35))*cx + (1-np.cos(np.deg2rad(35)))*cy]]), dtype='float')
b2 = np.array(([[np.cos(np.deg2rad(-125)), np.sin(np.deg2rad(-125)), (1 - np.cos(np.deg2rad(-125)))*cx - np.sin(np.deg2rad(-125))*cy],[-np.sin(np.deg2rad(-125)), np.cos(np.deg2rad(-125)), np.sin(np.deg2rad(-125))*cx + (1-np.cos(np.deg2rad(-125)))*cy]]), dtype='float')
c = np.array(([[0.4, 0, 0],[0, 0.4, 0]]), dtype='float')
#warping images with above-defined affine matrices
transimg = cv2.warpAffine(img, a, (img.shape[1], img.shape[0]))#translation
rotimg1 = cv2.warpAffine(img, b1, (img.shape[1], img.shape[0]))#35 degree rotaion
rotimg2 = cv2.warpAffine(img, b2, (img.shape[1], img.shape[0]))#-125 degree rotation
scaleimg = cv2.warpAffine(img, c, (img.shape[1], img.shape[0]))#scaling
#finally plotting images
plt.subplot(1,2,1)
plt.imshow(img, cmap='gray');plt.xticks([]);plt.yticks([]);plt.title('original image')
plt.subplot(1,2,2)
plt.imshow(rotimg1,cmap='gray');plt.xticks([]);plt.yticks([]);plt.title('rotated image (35)')
#%%
#Part 2
def he(im): #Histogram Equalization function
    N,_ = np.histogram(im,[i for i in range(0,257)])
    n = N/np.sum(N)
    z = np.zeros(n.shape)
    for i in range(len(n)):
        z[i] = 255*np.sum(n[:i])
    z = np.round(z)
    image = np.zeros(im.shape)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            image[i,j] = z[im[i,j]]
    image = np.uint8(image)
    return image

img = cv2.imread('60/2.png',0)
image = he(img)#histogram equalization

#creating plots
fig, axs = plt.subplots(2, 2, figsize=(9,9))

axs[0,0].imshow(img,cmap='gray', aspect = 'auto');axs[0,0].set_xticks([]);axs[0,0].set_yticks([]);axs[0,0].set_title('original image')
axs[0,1].imshow(image,cmap='gray', aspect = 'auto');axs[0,1].set_xticks([]);axs[0,1].set_yticks([]);axs[0,1].set_title('hist-equalised-image')
axs[1,0].hist(img.ravel(),256,[0,256])
axs[1,1].hist(image.ravel(),256,[0,256])
#%%
#Part 3
def noise(img, noise='gauss', var = 1):#noise function
    shape = img.shape
    sigma = var**0.5
    if noise == 'gauss':
        gauss = np.random.normal(0, sigma, shape)
        noisy_img = img + np.uint8(gauss)
        return np.uint8(noisy_img)
    else:
        noisy_img = np.copy(img)
        rand = np.random.randint(256, size = img.shape)
        u_limit = np.ceil(np.mean(rand)+np.std(rand)/(sigma+1e-07))
        l_limit = np.floor(np.mean(rand)-np.std(rand)/(sigma+1e-07))
        noisy_img[rand>u_limit] = 255
        noisy_img[rand<l_limit] = 0
        return np.uint8(noisy_img)
   
def filterr(img, filt = 'mean'):#filtering function
    shape = img.shape
    if filt == 'mean':
        lpf = (1/25)*np.array([[1,1,1,1,1],[1,1,1,1,1],[1,1,1,1,1],[1,1,1,1,1],[1,1,1,1,1]])
    elif filt == 'gauss':
        lpf = (1/256)*np.array([[1,4,6,4,1],[4,16,24,16,4],[6,24,36,24,6],[4,16,24,16,4],[1,4,6,4,1]])
    else:#median filtering. Padding 1 layer around the image in 'reflect' mode.
        new_img = cv2.copyMakeBorder(img, 1, 1, 1, 1, cv2.BORDER_REFLECT)
        for i in range(1,shape[0]+1):
            for j in range(1, shape[0]+1):    
                w = new_img[i-1:i+2,j-1:j+2].ravel()
                new_img[i,j] = np.median(w)
        n_img = new_img[1:shape[0]+1,1:shape[1]+1]
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        cl1 = clahe.apply(np.uint8(n_img))
        return np.uint8(cl1)
    result = signal.convolve2d(img,lpf,mode='same',boundary='symm')
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    cl1 = clahe.apply(np.uint8(result))
    return np.uint8(cl1)
    
img = cv2.imread('60/2.png',0)
var = 1.0
sp = noise(img,'gauss', var)#noise addition

#filtering
final1 = filterr(sp, filt='mean')
final2 = filterr(sp, filt='gauss')
final3 = filterr(sp, filt = 'sp')

#creating plots
fig, axs = plt.subplots(2, 5, figsize=(24, 8))

axs[0,0].imshow(img, cmap='gray', aspect = 'auto');axs[0,0].set_xticks([]);axs[0,0].set_yticks([]);axs[0,0].set_title('original')
axs[1,0].hist(img.ravel(),256,[0,256])

axs[0,1].imshow(sp, cmap='gray', aspect = 'auto');axs[0,1].set_xticks([]);axs[0,1].set_yticks([]);axs[0,1].set_title('Gauss (var=1)')
axs[1,1].hist(sp.ravel(),256,[0,256]);

axs[0,2].imshow(final1, cmap='gray', aspect = 'auto');axs[0,2].set_xticks([]);axs[0,2].set_yticks([]);axs[0,2].set_title('mean & CLAHE')
axs[1,2].hist(final1.ravel(),256,[0,256])

axs[0,3].imshow(final2, cmap='gray', aspect = 'auto');axs[0,3].set_xticks([]);axs[0,3].set_yticks([]);axs[0,3].set_title('gauss & CLAHE')
axs[1,3].hist(final2.ravel(),256,[0,256])

axs[0,4].imshow(final3, cmap='gray', aspect = 'auto');axs[0,4].set_xticks([]);axs[0,4].set_yticks([]);axs[0,4].set_title('median & CLAHE')
axs[1,4].hist(final3.ravel(),256,[0,256])

print('var: ', var)
print('noise: ', cv2.PSNR(img,sp))
print('mean: ', cv2.PSNR(img,final1))
print('gauss: ', cv2.PSNR(img,final2))
print('median: ', cv2.PSNR(img,final3))

    
