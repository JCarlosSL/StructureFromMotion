import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from scipy.ndimage.filters import convolve

def GaussianFunction(sigma,x,y):
    num = np.exp(-((x**2) + (y**2))/(2.0*(sigma**2)))
    den = 2.0*np.pi*(sigma**2)
    return num/den

def GaussianKernel(k=5,sigma=1):
    kernel=np.zeros((k,k),dtype=np.float32)
    if sigma==0 : return kernel
    m = n = k//2
    suma = 0
    for x in range(-m,m+1):
        for y in range(-n,n+1):
            value= GaussianFunction(sigma,x,y)
            kernel[x+2,y+2] = value
            suma+=value
    return kernel/suma

def GaussianFilter(img,kernel):
    img_conv = np.zeros_like(img,dtype=np.float32)
    if len(img.shape)==2:
        return convolve(img,kernel).astype(np.uint8)

    for c in range(3):
        img_conv[:,:,c] = convolve(img[:,:,c],kernel)

    return img_conv.astype(np.uint8)

def GaussianProcess(img,kernel_size,sigma):
    kernel = GaussianKernel(kernel_size,sigma)
    return GaussianFilter(img,kernel)

if __name__=='__main__':
    img =cv.imread('thresh3.png')

    print(kernel1)
    #Gimg = GaussianProcess(img,5,1)
    #Gim =cv.GaussianBlur(img,(5,5),1)
    #Cimg = cv.Canny(Gimg,100,200)
    #cv.imwrite('gimg.png',Gimg)


