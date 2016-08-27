# Import the necessary packages
import numpy as np
import cv2
import math
import Preprocess as pp
import Main
import os

import DetectChars
import DetectPlates
import PossiblePlate


def translate(image, x, y):
	# Define the translation matrix and perform the translation
	M = np.float32([[1, 0, x], [0, 1, y]])
	shifted = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))

	# Return the translated image
	return shifted

def rotate(image, angle, center = None, scale = 1.0):
	# Grab the dimensions of the image
	(h, w) = image.shape[:2]

	# If the center is None, initialize it as the center of
	# the image
	if center is None:
		center = (w / 2, h / 2)

	# Perform the rotation
	M = cv2.getRotationMatrix2D(center, angle, scale)
	rotated = cv2.warpAffine(image, M, (w, h))

	# Return the rotated image
	return rotated

def resize(image, width = None, height = None, inter = cv2.INTER_AREA):
	# initialize the dimensions of the image to be resized and
	# grab the image size
	dim = None

	(h, w) = image.shape[:2]

	# if both the width and height are None, then return the
	# original image
	if width is None and height is None:
		return image

	# check to see if the width is None
	if width is None:
		# calculate the ratio of the height and construct the
		# dimensions
		r = height / float(h)
		dim = (int(w * r), height)

	# otherwise, the height is None
	else:
		# calculate the ratio of the width and construct the
		# dimensions
		r = width / float(w)
		dim = (width, int(h * r))

	# resize the image
	resized = cv2.resize(image, dim, interpolation = inter)

	# return the resized image
	return resized

def transform (image):
  CAL_VAL = np.loadtxt("calibrated_value.txt")
  imheight = np.size(image, 0)
  imwidth = np.size(image, 1)
  M = getTransform (imwidth, imheight, CAL_VAL[2], CAL_VAL[3], CAL_VAL[4], CAL_VAL[5], CAL_VAL[6], CAL_VAL[7], CAL_VAL[8])
  transformed  = cv2.warpPerspective(image, M, (imwidth,imheight),cv2.INTER_CUBIC or cv2.WARP_INVERSE_MAP)
  return transformed

def detransform(image):
  CAL_VAL = np.loadtxt("calibrated_value.txt")
  imheight = np.size(image, 0)
  imwidth = np.size(image, 1)
  M = getTransform (imwidth, imheight, (0-CAL_VAL[2]), (0-CAL_VAL[3]), (0-CAL_VAL[4]), (0-CAL_VAL[5]), (0-CAL_VAL[6]), (1-CAL_VAL[7]), (1-CAL_VAL[8]))
  #M = getTransform (imwidth, imheight, 0.0, 0.0, 0.0, 0, 0, 1.0,1.0)
  detransformed  = cv2.warpPerspective(image, M, (imwidth,imheight),cv2.INTER_CUBIC or cv2.WARP_INVERSE_MAP)
  return detransformed

def getTransform (w, h, rotationx, rotationy, rotationz, panX, panY, stretchX, dist):
    alpha = rotationx;
    beta = rotationy;
    gamma = rotationz;
    f = 1.0;
    A1 = np.matrix([[1, 0, -w/2], [0, 1, -h/2],[0, 0,0],[0, 0,1]])
    #print(A1)
    A2 = np.matrix([[f, 0, w/2, 0], [0, f, h/2, 0],[0, 0, 1, 0]])
    #print(A2)
    Rx = np.matrix([[1, 0, 0, 0],[0, math.cos(alpha), -(math.sin(alpha)), 0],[0, math.sin(alpha),  math.cos(alpha), 0],[0, 0, 0, 1]])
    #print(Rx)
    Ry = np.matrix([[math.cos(beta), 0, math.sin(beta), 0],[0, 1, 0, 0],[-(math.sin(beta)), 0,  math.cos(beta), 0],[0, 0, 0, 1]])
    #print(Ry)
    Rz = np.matrix([[math.cos(gamma), -(math.sin(gamma)), 0, 0],[math.sin(gamma), math.cos(gamma), 0, 0],[0, 0, 1, 0],[0, 0, 0, 1]])
    #print(Rz)
    R = Rx*Ry*Rz
    #print(R)
    T = np.matrix([[stretchX, 0, 0, panX],[0, 1, 0, panY],[0, 0, 1, dist],[0, 0, 0, 1]])
    #print(T)
    M = A2*(T*(R*A1))
    #print(M)
    return M
