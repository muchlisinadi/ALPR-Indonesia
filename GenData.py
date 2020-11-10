# GenData.py
import argparse
import os
import sys

import cv2
import numpy as np

# module level variables ##########################################################################
MIN_CONTOUR_AREA = 100

RESIZED_IMAGE_WIDTH = 20
RESIZED_IMAGE_HEIGHT = 30


###################################################################################################
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-d", "--image_train",
                    help="path for the images that you're going to train")
    args = vars(ap.parse_args())
    if args.get("image", True):
        imgTrainingNumbers = cv2.imread(args["image_train"])  # read in training numbers image
        if imgTrainingNumbers is None:
            print
            "error: image not read from file \n\n"  # print error message to std out
            os.system("pause")  # pause so user can see error message
            return
    else:
        print("Please add -d or --image_train argument")

    imgGray = cv2.cvtColor(imgTrainingNumbers, cv2.COLOR_BGR2GRAY)  # get grayscale image
    imgBlurred = cv2.GaussianBlur(imgGray, (5, 5), 0)  # blur

    # filter image from grayscale to black and white
    imgThresh = cv2.adaptiveThreshold(imgBlurred,  # input image
                                      255,  # make pixels that pass the threshold full white
                                      cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                      # use gaussian rather than mean, seems to give better results
                                      cv2.THRESH_BINARY_INV,
                                      # invert so foreground will be white, background will be black
                                      11,  # size of a pixel neighborhood used to calculate threshold value
                                      2)  # constant subtracted from the mean or weighted mean

    cv2.imshow("imgThresh", imgThresh)  # show threshold image for reference

    imgThreshCopy = imgThresh.copy()  # make a copy of the thresh image, this in necessary b/c findContours modifies the image

    imgContours, npaContours, npaHierarchy = cv2.findContours(imgThreshCopy,
                                                              # input image, make sure to use a copy since the function will modify this image in the course of finding contours
                                                              cv2.RETR_EXTERNAL,  # retrieve the outermost contours only
                                                              cv2.CHAIN_APPROX_SIMPLE)  # compress horizontal, vertical, and diagonal segments and leave only their end points

    # declare empty numpy array, we will use this to write to file later
    # zero rows, enough cols to hold all image data
    npaFlattenedImages = np.empty((0, RESIZED_IMAGE_WIDTH * RESIZED_IMAGE_HEIGHT))

    intClassifications = []  # declare empty classifications list, this will be our list of how we are classifying our chars from user input, we will write to file at the end

    # possible chars we are interested in are digits 0 through 9, put these in list intValidChars
    intValidChars = [ord('0'), ord('1'), ord('2'), ord('3'), ord('4'), ord('5'), ord('6'), ord('7'), ord('8'), ord('9'),
                     ord('a'), ord('b'), ord('c'), ord('d'), ord('e'), ord('f'), ord('g'), ord('h'), ord('i'), ord('j'),
                     ord('k'), ord('l'), ord('m'), ord('n'), ord('o'), ord('p'), ord('q'), ord('r'), ord('s'), ord('t'),
                     ord('u'), ord('v'), ord('w'), ord('x'), ord('y'), ord('z')]
    for npaContour in npaContours:  # for each contour
        if cv2.contourArea(npaContour) > MIN_CONTOUR_AREA:  # if contour is big enough to consider
            [intX, intY, intW, intH] = cv2.boundingRect(npaContour)  # get and break out bounding rect

            # draw rectangle around each contour as we ask user for input
            cv2.rectangle(imgTrainingNumbers,  # draw rectangle on original training image
                          (intX, intY),  # upper left corner
                          (intX + intW, intY + intH),  # lower right corner
                          (0, 0, 255),  # red
                          2)  # thickness

            imgROI = imgThresh[intY:intY + intH, intX:intX + intW]  # crop char out of threshold image
            imgROIResized = cv2.resize(imgROI, (RESIZED_IMAGE_WIDTH,
                                                RESIZED_IMAGE_HEIGHT))  # resize image, this will be more consistent for recognition and storage

            cv2.imshow("imgROI", imgROI)  # show cropped out char for reference
            cv2.imshow("imgROIResized", imgROIResized)  # show resized image for reference
            cv2.imshow("training_numbers.png",
                       imgTrainingNumbers)  # show training numbers image, this will now have red rectangles drawn on it

            intChar = cv2.waitKey(0)  # get key press

            if intChar == 27:  # if esc key was pressed
                sys.exit()  # exit program
            elif intChar in intValidChars:  # else if the char is in the list of chars we are looking for . . .

                intClassifications.append(
                    intChar)  # append classification char to integer list of chars (we will convert to float later before writing to file)

                npaFlattenedImage = imgROIResized.reshape((1,
                                                           RESIZED_IMAGE_WIDTH * RESIZED_IMAGE_HEIGHT))  # flatten image to 1d numpy array so we can write to file later
                npaFlattenedImages = np.append(npaFlattenedImages, npaFlattenedImage,
                                               0)  # add current flattened impage numpy array to list of flattened image numpy arrays
            # end if
        # end if
    # end for

    fltClassifications = np.array(intClassifications,
                                  np.float32)  # convert classifications list of ints to numpy array of floats

    npaClassifications = fltClassifications.reshape(
        (fltClassifications.size, 1))  # flatten numpy array of floats to 1d so we can write to file later

    print
    "\n\ntraining complete !!\n"

    np.savetxt("classifications.txt", npaClassifications)  # write flattened images to file
    np.savetxt("flattened_images.txt", npaFlattenedImages)
    changeCaption()  #

    cv2.destroyAllWindows()  # remove windows from memory

    return


###################################################################################################
def changeCaption():
    data = np.loadtxt("classifications.txt")
    i = 0
    for a in data:
        a = int(round(a))
        if (a == ord('a')):
            data[i] = ord('A')
        if (a == ord('b')):
            data[i] = ord('B')
        if (a == ord('c')):
            data[i] = ord('C')
        if (a == ord('d')):
            data[i] = ord('D')
        if (a == ord('e')):
            data[i] = ord('E')
        if (a == ord('f')):
            data[i] = ord('F')
        if (a == ord('g')):
            data[i] = ord('G')
        if (a == ord('h')):
            data[i] = ord('H')
        if (a == ord('i')):
            data[i] = ord('I')
        if (a == ord('j')):
            data[i] = ord('J')
        if (a == ord('k')):
            data[i] = ord('K')
        if (a == ord('l')):
            data[i] = ord('L')
        if (a == ord('m')):
            data[i] = ord('M')
        if (a == ord('n')):
            data[i] = ord('N')
        if (a == ord('o')):
            data[i] = ord('O')
        if (a == ord('p')):
            data[i] = ord('P')
        if (a == ord('q')):
            data[i] = ord('Q')
        if (a == ord('r')):
            data[i] = ord('R')
        if (a == ord('s')):
            data[i] = ord('S')
        if (a == ord('t')):
            data[i] = ord('T')
        if (a == ord('u')):
            data[i] = ord('U')
        if (a == ord('v')):
            data[i] = ord('V')
        if (a == ord('w')):
            data[i] = ord('W')
        if (a == ord('x')):
            data[i] = ord('X')
        if (a == ord('y')):
            data[i] = ord('Y')
        if (a == ord('z')):
            data[i] = ord('Z')
        i = i + 1

    # fltClassifications = np.array(intClassifications, np.float32)
    hasil = np.array(data, np.float32)  # convert classifications list of ints to numpy array of floats
    npaClassifications = hasil.reshape((hasil.size, 1))

    np.savetxt("classifications.txt", npaClassifications)
    # print("char was change to caption !")


if __name__ == "__main__":
    main()
# end if
