import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from gaussfilter import GaussianProcess as gf 

OCTAVE_DIM = 4

def ScaleDownImageByHalf(image):
    scaleDownImage = image[1::2, 1::2]
    return(scaleDownImage)

def GetSigmaMatrix():
    sigmaMatrix = np.array([[0.7071,1.0,1.4142,2.0,2.8284],
                            [1.4142,2.0,2.8284,4.0,5.6568],
                            [2.8284,4.0,5.6568,8.0,11.3137],
                            [5.6568,8.0,11.3137,16.0,22.6274]])
    return sigmaMatrix 

def GetOctave(image,sigmas,ksize=5):
    octave = []
    for i in range(5):
        octave.append(gf(image,ksize,sigmas[i]))
    return octave

def GenerateOctaves(image):
    sigmaMatrix = GetSigmaMatrix()
    octaves = []
    for i in range(OCTAVE_DIM):
        octave = GetOctave(image,sigmaMatrix[i])
        octaves.append(octave)
        image = ScaleDownImageByHalf(image)
    return octaves

def ComputeDoG(octave):
    dogs = []
    for i in range(4):
        dog = octave[i]-octave[i+1]
        dogs.append(dog)
    return dogs

def GenerateDoG(octaves):
    dogs = []
    for i in range(OCTAVE_DIM):
        dog = ComputeDoG(octaves[i])
        dogs.append(dog)
    return dogs

if __name__=='__main__':
    img = cv.imread('thresh3.png',0)
    octaves = GenerateOctaves(img)
    dogs = GenerateDoG(octaves)

    print(len(dogs[0]))
    #cv.imwrite('img.png',dogs[3][3])
