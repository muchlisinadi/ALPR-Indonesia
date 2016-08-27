# Preprocess.py

import cv2
import numpy as np
import math

# module level variables ##########################################################################
CAL_VAL = np.loadtxt("calibrated_value.txt")
(w ,h, rotationx, rotationy, rotationz, panX, panY, stretchX, dist, G_S_F_W, G_S_F_H, A_T_B, A_T_W, T_V, Xtrans, Ytrans) = np.loadtxt("calibrated_value.txt")
GAUSSIAN_SMOOTH_FILTER_SIZE = (int(G_S_F_W), int(G_S_F_H)) # last best = 3,3
ADAPTIVE_THRESH_BLOCK_SIZE = int(A_T_B) #19 , last best = 19
ADAPTIVE_THRESH_WEIGHT = int(A_T_W) #9, last best = 11
THRESHOLD_VALUE = int(T_V)
##
###################################################################################################
def preprocess(imgOriginal):
    #coba = cv2.cvtColor(imgOriginal, cv2.COLOR_BGR2HSV)
    #cv2.imshow("hsv", coba )
    imgGrayscale = extractValue(imgOriginal)
    #cv2.imshow("imgGrayscale", imgGrayscale )

    imgGrayscale = np.invert(imgGrayscale) # last best use this
    #cv2.imshow("invert", imgGrayscale )
    imgMaxContrastGrayscale = maximizeContrast(imgGrayscale)
    #cv2.imshow("imgMaxContrastGrayscale", imgMaxContrastGrayscale )
    #imgMaxContrastGrayscale = np.invert(imgMaxContrastGrayscale)
    height, width = imgGrayscale.shape

    imgBlurred = np.zeros((height, width, 1), np.uint8)
    #cv2.imshow("c_3", imgBlurred )

    imgBlurred = cv2.GaussianBlur(imgMaxContrastGrayscale, GAUSSIAN_SMOOTH_FILTER_SIZE, 0)
    #cv2.imshow("imgBlurred", imgBlurred )
    #imgBlurred = np.invert(imgBlurred)
    imgThresh = cv2.adaptiveThreshold(imgBlurred, THRESHOLD_VALUE , cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, ADAPTIVE_THRESH_BLOCK_SIZE, ADAPTIVE_THRESH_WEIGHT)
    #imgThresh = np.invert(imgThresh)
    #cv2.imshow("cobaaa", imgThresh)

    return imgGrayscale, imgThresh
# end function

###################################################################################################
def extractValue(imgOriginal):
    height, width, numChannels = imgOriginal.shape

    imgHSV = np.zeros((height, width, 3), np.uint8)

    imgHSV = cv2.cvtColor(imgOriginal, cv2.COLOR_BGR2HSV)

    imgHue, imgSaturation, imgValue = cv2.split(imgHSV)

    return imgValue
# end function

###################################################################################################
def maximizeContrast(imgGrayscale):

    height, width = imgGrayscale.shape

    imgTopHat = np.zeros((height, width, 1), np.uint8)
    imgBlackHat = np.zeros((height, width, 1), np.uint8)

    structuringElement = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

    imgTopHat = cv2.morphologyEx(imgGrayscale, cv2.MORPH_TOPHAT, structuringElement)
    imgBlackHat = cv2.morphologyEx(imgGrayscale, cv2.MORPH_BLACKHAT, structuringElement)

    imgGrayscalePlusTopHat = cv2.add(imgGrayscale, imgTopHat)
    imgGrayscalePlusTopHatMinusBlackHat = cv2.subtract(imgGrayscalePlusTopHat, imgBlackHat)

    return imgGrayscalePlusTopHatMinusBlackHat
# end function
