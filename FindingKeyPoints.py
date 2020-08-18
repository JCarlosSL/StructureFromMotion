import numpy as np
import cv2 as cv

from constant import *
from keypoints_class import *
from differentGaussian import *

def ComputeKeypoints(dog, nro_dog):
    M,N = dog.shape[0],dog.shape[1]
    keypoint=[]
    for k in range(1,3):
        for i in range(1,M-1):
            for j in range(1,N-1):
                if np.abs(dog[i,j,k])< THRESHOLD:
                    continue

                values = np.concatenate((dog[i-1:i+2,j-1:j+2,k-1],np.concatenate((
                    dog[i-1:i+2,j-1:j+2,k],dog[i-1:i+2,j-1:j+2,k+1]))))
                max_value = np.max(values)
                min_value = np.min(values)

                if (dog[i,j,k]==max_value) or (dog[i,j,k] == min_value):
                    # find fisrt derivatives approximated as difference
                    dx = (dog[i,j+1,k]-dog[i,j-1,k])*0.5/255
                    dy = (dog[i+1,j,k]-dog[i-1,j,k])*0.5/255
                    ds = (dog[i,j,k+1]-dog[i,j,k-1])*0.5/255

                    dD = np.matrix([[dx],[dy],[ds]])

                    dxx = (dog[i,j+1,k]+dog[i,j-1,k]-2*dog[i,j,k])*1.0/255
                    dyy = (dog[i+1,j,k]+dog[i-1,j,k]-2*dog[i,j,k])*1.0/255
                    dss = (dog[i,j,k+1]+dog[i,j,k-1]-2*dog[i,j,k])*1.0/255

                    dxy = (dog[i+1,j+1,k]-dog[i+1,j-1,k]
                            -dog[i-1,j+1,k]+dog[i-1,j-1,k])*0.25/255
                    dxs = (dog[i,j+1,k+1]-dog[i,j-1,k+1]
                            -dog[i,j+1,k-1]+dog[i,j-1,k-1])*0.25/255
                    dys = (dog[i+1,j,k+1]-dog[i-1,j,k+1]
                            -dog[i+1,j,k-1]+dog[i-1,j,k-1])*0.25/255
                    H = np.matrix([[dxx,dxy,dxs],[dxy,dyy,dys],[dxs,dys,dss]])

                    x_hat = np.linalg.lstsq(H,dD,rcond=None)[0]
                    D_x_hat = dog[i,j,k]+0.5*np.dot(dD.transpose(),x_hat)

                    alpha = ((RADIUS_OF_CURVATURE +1)**2)
                    trace_H_sq = (dxx + dyy)**2
                    det_H = dxx*dyy-(dxy**2)

                    if (trace_H_sq*RADIUS_OF_CURVATURE < alpha*det_H) and \
                            (np.abs(x_hat[0]) < 0.5) and \
                            (np.abs(x_hat[1]) < 0.5) and \
                            (np.abs(x_hat[2]) < 0.5) and \
                            (np.abs(D_x_hat) > 0.03):
                        temp_keypoint = KEYPOINT(i=i,j=j,octave=nro_dog,DoG=k,
                                x=j+x_hat[0],y=i+x_hat[1])
                        keypoint.append(temp_keypoint)

def GenerateKeyPoint(dogs):
    keypoints = []
    for i in range(OCTAVE_DIM):
        keypoints.append(ComputeKeypoints(np.array(dogs[i]),i+1))

    return keypoints

if __name__=='__main__':
    img = cv.imread('sample-input.jpeg',0)
    octaves = generateScaleSpace(img)
    dogs = generateDoG(octaves)

    keypoints = GenerateKeyPoint(dogs)
