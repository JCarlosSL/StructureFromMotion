import numpy as np
from scipy.ndimage import gaussian_filter
import cv2 as cv

from constant import *
from keypoints_class import *

def downsampled_images(image,rate=2):
    return image[0::rate,0::rate]

def gaussian_blur(image,sd):
    return gaussian_filter(image,sigma=sd)

def getSigma():
    return [SIGMA,SIGMA*2,SIGMA*4,SIGMA*8]

def computeScaleSpace(image,sigma,k):
    octave_dim = (image.shape[0],image.shape[1],OCTAVE_DIMENSION)
    octave = np.zeros(octave_dim)
    for i in range(0, OCTAVE_DIMENSION):
        octave[:,:,i] = gaussian_blur(image,sigma * ( k ** i ))
    return octave

def generateScaleSpace(image):
    sigmas=getSigma()
    octaves = []
    s=OCTAVE_DIMENSION - 3
    k=np.power(2,1/s) 
    for i in range(OCTAVE_DIMENSION-1):
        octave = computeScaleSpace(image,sigmas[i],k)
        octaves.append(octave)
        image=downsampled_images(image)

    return octaves

def computeDoG(scale):
    dog_dim = (scale.shape[0],scale.shape[1],OCTAVE_DIMENSION-1)
    dog=np.zeros(dog_dim)
    for i in range(OCTAVE_DIMENSION-1):
        dog[:,:,i] = scale[:,:,i+1]-scale[:,:,i]
    return dog

def generateDoG(octaves):
    dogs = []
    for i in range(OCTAVE_DIMENSION-1):
        dogs.append(computeDoG(octaves[i]))
    return dogs

if __name__ == '__main__':
    image = cv.imread('sample-input.jpeg',0)
    octaves = generateScaleSpace(image)
    dogs = generateDoG(octaves)
