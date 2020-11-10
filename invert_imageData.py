# GenData.py

import argparse
import os

import cv2
import numpy as np


# module level variables ##########################################################################


###################################################################################################
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-d", "--image_train",
                    help="path for the images that you're going to invert")
    args = vars(ap.parse_args())
    if args.get("image", True):
        imgTrainingNumbers = cv2.imread(args["image_train"])  # read in training numbers image
        if imgTrainingNumbers is None:
            print("error: image not read from file \n\n")  # print error message to std out
            os.system("pause")  # pause so user can see error message
            return
    else:
        print("Please add -d or --image_train argument")

    imgGray = cv2.cvtColor(imgTrainingNumbers, cv2.COLOR_BGR2GRAY)  # get grayscale image
    imgBlurred = cv2.GaussianBlur(imgGray, (5, 5), 0)  # blur

    # filter image from grayscale to black and white
    imgThresh = cv2.adaptiveThreshold(imgBlurred,  # input image
                                      0,  # make pixels that pass the threshold full white
                                      cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                      # use gaussian rather than mean, seems to give better results
                                      cv2.THRESH_BINARY_INV,
                                      # invert so foreground will be white, background will be black
                                      11,  # size of a pixel neighborhood used to calculate threshold value
                                      2)  # constant subtracted from the mean or weighted mean

    imgTrainingNumbers = np.invert(imgTrainingNumbers)
    cv2.imwrite("invert_" + args["image_train"], imgTrainingNumbers)
    cv2.imwrite("imgThresh_" + args["image_train"], imgThresh)
    return


###################################################################################################
if __name__ == "__main__":
    main()
# end if
