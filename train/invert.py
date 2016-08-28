# GenData.py

import sys
import numpy as np
import cv2
import os

# module level variables ##########################################################################


###################################################################################################
def main():
    imgTrainingNumbers = cv2.imread("dddd.png")            # read in training numbers image

    if imgTrainingNumbers is None:                          # if image was not read successfully
        print "error: image not read from file \n\n"        # print error message to std out
        os.system("pause")                                  # pause so user can see error message
        return                                              # and exit function (which exits program)
    
    imgGray = cv2.cvtColor(imgTrainingNumbers, cv2.COLOR_BGR2GRAY)          # get grayscale image
    imgBlurred = cv2.GaussianBlur(imgGray, (5,5), 0)                        # blur

                                                        # filter image from grayscale to black and white
    imgThresh = cv2.adaptiveThreshold(imgBlurred,                           # input image
                                      0,                                  # make pixels that pass the threshold full white
                                      cv2.ADAPTIVE_THRESH_GAUSSIAN_C,       # use gaussian rather than mean, seems to give better results
                                      cv2.THRESH_BINARY_INV,                # invert so foreground will be white, background will be black
                                      11,                                   # size of a pixel neighborhood used to calculate threshold value
                                      2)                                    # constant subtracted from the mean or weighted mean

    imgTrainingNumbers = np.invert(imgTrainingNumbers)
    cv2.imwrite("invers2.jpg",imgTrainingNumbers)
    cv2.imwrite("imgThresh2.jpg",imgThresh)
    return

###################################################################################################
if __name__ == "__main__":
    main()
# end if




