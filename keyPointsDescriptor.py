from differentGaussian import *
from FindingKeyPoints import *
from numpy.linalg import norm
import cv2

#def compute_magnitude_angle(scale1, scale2, scale3, scale4):
def compute_magnitude_angle(scale):
	#scale=[scale1,scale2,scale3,scale4]
	magnitude = [0,0,0,0]
	theta = [0,0,0,0]
	# scale 1
	for i in range(len(scale)):
		(M, N) = (scale[i].shape[0], scale[i].shape[1])
		scale_x_1p = np.zeros_like(scale[i][:, :, :OCTAVE_DIMENSION - 3])
		scale_x_1n = np.zeros_like(scale[i][:, :, :OCTAVE_DIMENSION - 3])
		scale_y_1p = np.zeros_like(scale[i][:, :, :OCTAVE_DIMENSION - 3])
		scale_y_1n = np.zeros_like(scale[i][:, :, :OCTAVE_DIMENSION - 3])

		scale_x_1n[:, 1:, :] = scale[i][:, :N-1, 1:OCTAVE_DIMENSION-2]
		scale_x_1p[:, :N-1, :] = scale[i][:, 1:, 1:OCTAVE_DIMENSION-2]
		scale_y_1n[1:, :, :] = scale[i][:M-1, :, 1:OCTAVE_DIMENSION-2]
		scale_y_1p[:M-1, :, :] = scale[i][1:, :, 1:OCTAVE_DIMENSION-2]

		magnitude[i] = np.sqrt(np.square(scale_x_1p - scale_x_1n) + np.square(scale_y_1p - scale_y_1n))
		theta[i] = np.arctan2(scale_y_1p - scale_y_1n, scale_x_1p - scale_x_1n) * 180 / np.pi
		theta[i] = np.mod(theta[i] + 360, 360 * np.ones_like(theta[i]))
	#return magnitude[0], theta[0], magnitude[1], theta[1], magnitude[2], theta[2], magnitude[3], theta[3]
	return magnitude, theta

def compute_histogram(magnitude, theta):
	histogram = np.zeros(36)

	for i in range(0, theta.shape[0]):
		for j in range(0, theta.shape[1]):
			histogram[np.uint8(theta[i, j])] = histogram[np.uint8(theta[i, j])] + magnitude[i, j]

	return histogram


def compute_descriptor(scale1, M1, T1, M2, T2, M3, T3, M4, T4, keypoints):
	magnitude = np.zeros((scale1.shape[0], scale1.shape[1], 8))
	orientation = np.zeros((scale1.shape[0], scale1.shape[1], 8))

	for i in range(0, 2):
		magnitude[:, :, i] = (M1[:, :, i]).astype(float)
		orientation[:, :, i] = (T1[:, :, i]).astype(int)
		magnitude[:, :, i + 2] = np.resize(M2[:, :, i], (scale1.shape[0], scale1.shape[1])).astype(int)
		orientation[:, :, i + 2] = np.resize(T2[:, :, i], (scale1.shape[0], scale1.shape[1])).astype(int)
		magnitude[:, :, i + 4] = np.resize(M3[:, :, i], (scale1.shape[0], scale1.shape[1])).astype(int)
		orientation[:, :, i + 4] = np.resize(T3[:, :, i], (scale1.shape[0], scale1.shape[1])).astype(int)
		magnitude[:, :, i + 6] = np.resize(M4[:, :, i], (scale1.shape[0], scale1.shape[1])).astype(int)
		orientation[:, :, i + 6] = np.resize(T4[:, :, i], (scale1.shape[0], scale1.shape[1])).astype(int)

	descriptors = np.zeros((len(keypoints), 128))
	k = np.sqrt(2)
	kvectotal = np.uint8(np.array([SIGMA, SIGMA * k, SIGMA * (k ** 2), SIGMA * (k ** 3), SIGMA * (k ** 4), SIGMA * (k ** 5),
						  SIGMA * (k ** 6), SIGMA * (k ** 7)]) * 1000)

	for i in range(0, len(keypoints)):
		x0 = np.int(keypoints[i].x)
		y0 = np.int(keypoints[i].y)
		value = np.uint8(keypoints[i].sigma * 1000)
		scale_idx = np.int8(np.argwhere(kvectotal == value))[0][0]

		gaussian_window = gaussian_blur(magnitude[x0-16:x0+16, y0-16:y0+16, scale_idx], keypoints[i].sigma)

		if type(gaussian_window.size) == int and gaussian_window.size < 1024:
			continue

		for x in range(-8, 8):
			for y in range(-8, 8):
				theta = keypoints[i].angle * np.pi / 180.0

				xrot = np.int(np.round((np.cos(theta) * x) - (np.sin(theta) * y)))
				yrot = np.int(np.round((np.sin(theta) * x) + (np.cos(theta) * y)))

				x_ = np.int8(x0 + xrot)
				y_ = np.int8(y0 + yrot)

				weight = gaussian_window[xrot+8, yrot+8]

				angle = orientation[x_, y_, scale_idx] - keypoints[i].angle
				angle = np.int8(angle/10)

				if angle < 0:
					angle = 36 + angle

				bin_idx = np.clip(np.floor((8.0 / 36) * angle), 0, 7).astype(int)
				bin_idx = bin_idx[0]
				descriptors[i, 32 * int((x + 8) / 4) + 8 * int((y + 8) / 4) + bin_idx] += weight

		descriptors[i, :] = descriptors[i, :] / norm(descriptors[i, :])
		descriptors[i, :] = np.clip(descriptors[i, :], 0, 0.2)

		descriptors[i, :] = descriptors[i, :] / norm(descriptors[i, :])

	return descriptors
	# return np.uint8(descriptors)


#def compute_final_keypoints_descriptors(keyset1, keyset2, keyset3, keyset4, scale1, scale2, scale3, scale4):
def compute_final_keypoints_descriptors(keyset, scale):
	keypoint_list = []
	descriptor_list = []

	# Compute magnitude and orientation of all points in scale space
	(M, T_T) = compute_magnitude_angle(scale)#1, scale2, scale3, scale4)

	# Covert 0 to 360 degrees into 36 bins
	T=[]
	for i in range(len(T_T)):
		T.append( np.floor(T_T[i] / 10))

	new_keypoints = [0,0,0,0]
	#print(keyset)
	for i in range(len(keyset)):
		# Scale 1
		new_keypoints[i] = []
		for key in keyset[i]:
			(x, y) = key.j, key.i
			k = np.uint8(key.DoG - 1)
			minx = np.uint8(min(8, x))
			miny = np.uint8(min(8, y))

			theta_neighbourhood = T[i][x - minx: x + minx + 1, y - miny: y + miny + 1, k]
			magnitude_neighbourhood = M[i][x - minx: x + minx + 1, y - miny: y + miny + 1, k]
			magnitude_neighbourhood = gaussian_blur(magnitude_neighbourhood, 1.5 * SIGMA * (np.sqrt(2))**(k+1))

			histogram = compute_histogram(magnitude_neighbourhood, theta_neighbourhood)
			histogram_sorted = histogram.copy()
			histogram_sorted.sort()

			key.sigma = SIGMA * (np.sqrt(2)) ** (k + 1)

			if histogram_sorted[-2] > 0.8 * histogram_sorted[-1]:
				key.angle = np.argwhere(histogram == histogram_sorted[-1])[0] * 10
				new_keypoints[i].append(key)

				for h in range(2, 36):
					if histogram_sorted[-h] > 0.8 * histogram_sorted[-1]:
						key.angle = np.argwhere(histogram == histogram_sorted[-h])[0] * 10
						new_keypoints[i].append(key)
					else:
						break
			else:
				key.angle = np.argwhere(histogram == histogram_sorted[-1])[0] * 10
				new_keypoints[i].append(key)

	new_keypoints = normalize_keypoints(new_keypoints[0], new_keypoints[1], new_keypoints[2], new_keypoints[3])
	# Remove duplicate keys
	keypoint_list = list(set(new_keypoints))

	for k in keypoint_list:
		if k.i < 16 or k.j < 16 or k.i > (scale[0].shape[0] - 17) or k.j > (scale[0].shape[1] - 17):
			keypoint_list.remove(k)

	descriptor_list = compute_descriptor(scale[0], M[0], T_T[0], M[1], T_T[1], M[2], T_T[2], M[3], T_T[3], keypoint_list)

	return keypoint_list, descriptor_list


def convert_KEYPOINT_to_Keypoint(source):
	dest = []

	for k in source:
		key = cv2.KeyPoint(k.x, k.y, k.DoG)
		dest.append(key)

	return dest


def normalize_keypoints(keyset1, keyset2, keyset3, keyset4):
	keyset = []

	for k in keyset1:
		keyset.append(k)

	for k in keyset2:
		k.i = 2 * k.i
		k.j = 2 * k.j
		k.x = 2 * k.x
		k.y = 2 * k.y
		keyset.append(k)

	for k in keyset3:
		k.i = 4 * k.i
		k.j = 4 * k.j
		k.x = 4 * k.x
		k.y = 4 * k.y
		keyset.append(k)

	for k in keyset4:
		k.i = 8 * k.i
		k.j = 8 * k.j
		k.x = 8 * k.x
		k.y = 8 * k.y
		keyset.append(k)

	return keyset


