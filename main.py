from differentGaussian import *
from FindingKeyPoints import *
from keyPointsDescriptor import *
import cv2
import matplotlib.pyplot as plt

#image = cv.imread('sample-input.jpeg',0)
def processimage(path):
	image = cv.imread(path,0)
	octaves = generateScaleSpace(image)
	dogs = generateDoG(octaves)
	#print(dogs)
	initkeypoints = GenerateKeyPoint(dogs)
	#print(initkeypoints)
	(keypoints, descriptor) = compute_final_keypoints_descriptors(initkeypoints, octaves)
	#print(keypoints)
	#print(descriptor)
	keypoints = convert_KEYPOINT_to_Keypoint(keypoints)
	output_image = cv2.drawKeypoints(image, keypoints, cv2.DRAW_MATCHES_FLAGS_DEFAULT)
	# Convert output image to RGB space
	output_image = cv2.cvtColor(output_image,cv2.COLOR_BGR2RGB)
	# Display output image
	plt.imshow(output_image)
	plt.show()
	return (keypoints,descriptor)

keyp = []
imgs = ['img1.jpeg',
		'img2.jpeg',
		'img3.jpeg',
		'img4.jpeg',]
for i in range(len(imgs)):
	keyp.append(processimage(imgs[i]))
	print(len(keyp[i][0]))

