# USAGE
# python Main.py --image Sample/s1.jpg  @ untuk file gambar
# python Main.py --video Sample/sv1.mp4  @ untuk file video
# python Main.py @ untuk cam


import argparse
import os

import cv2

import Calibration as cal
import DetectChars
import DetectPlates
import Preprocess as pp
import imutils

# Module level variables for image ##########################################################################

SCALAR_BLACK = (0.0, 0.0, 0.0)
SCALAR_WHITE = (255.0, 255.0, 255.0)
SCALAR_YELLOW = (0.0, 255.0, 255.0)
SCALAR_GREEN = (0.0, 255.0, 0.0)
SCALAR_RED = (0.0, 0.0, 255.0)
N_VERIFY = 5  # number of verification


def main():
    # argument for input video/image/calibration
    ap = argparse.ArgumentParser()
    ap.add_argument("-v", "--video", help="Path to video file")
    ap.add_argument("-i", "--image", help="Path to the image")
    ap.add_argument("-c", "--calibration", help="image or video or camera")
    args = vars(ap.parse_args())


    img_original_scene = None
    loop = None
    camera = None

    # if -c assigned, calibrate the angle of camera or video
    if args.get("calibration", True):
        img_original_scene = cv2.imread(args["calibration"])
        if img_original_scene is None:
            print("Please check again the path of image or argument !")
        img_original_scene = imutils.resize(img_original_scene, width=720)
        cal.calibration(img_original_scene)
        return
    else:  # run video / image / cam
        if args.get("video", True):
            camera = cv2.VideoCapture(args["video"])
            if camera is None:
                print("Please check again the path of video or argument !")
            loop = True

        elif args.get("image", True):
            img_original_scene = cv2.imread(args["image"])
            if img_original_scene is None:
                print("Please check again the path of image or argument !")
                loop = False
        else:
            camera = cv2.VideoCapture(0)
            loop = True

    # Load and check KNN Model
    assert DetectChars.loadKNNDataAndTrainKNN(), "KNN can't be loaded !"

    save_number = 0
    prev_license = ""
    licenses_verify = []

    # Looping for Video
    while loop:
        # grab the current frame
        (grabbed, frame) = camera.read()
        if args.get("video") and not grabbed:
            break

        # resize the frame and preprocess
        img_original_scene = imutils.resize(frame, width=620)
        _, img_thresh = pp.preprocess(img_original_scene)

        # Show the preprocess result
        cv2.imshow("threshold", img_thresh)

        # Get the license in frame
        img_original_scene = imutils.transform(img_original_scene)
        img_original_scene, new_license = searching(img_original_scene, loop)

        # only save 5 same license each time (verification)
        if new_license == "":
            print("no characters were detected\n")
        else:
            if len(licenses_verify) == N_VERIFY and len(set(licenses_verify)) == 1:
                if prev_license == new_license:
                    print(f"still = {prev_license}\n")
                else:
                    # show and save verified plate
                    print(f"A new license plate read from image = {new_license} \n")
                    cv2.imshow(new_license, img_original_scene)
                    file_name = f"hasil/{new_license}.png"
                    cv2.imwrite(file_name, img_original_scene)
                    prev_license = new_license
                    licenses_verify = []
            else:
                if len(licenses_verify) == N_VERIFY:
                    # drop first if reach the N_VERIFY
                    licenses_verify = licenses_verify[1:]
                licenses_verify.append(new_license)

        # add text and rectangle, just for information and bordering
        cv2.putText(img_original_scene, "Press 's' to save frame to be 'save.png', for calibrating", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, bottomLeftOrigin=False)
        cv2.rectangle(img_original_scene,
                      ((img_original_scene.shape[1] // 2 - 230), (img_original_scene.shape[0] // 2 - 80)),
                      ((img_original_scene.shape[1] // 2 + 230), (img_original_scene.shape[0] // 2 + 80)), SCALAR_GREEN,
                      3)
        cv2.imshow("imgOriginalScene", img_original_scene)

        key = cv2.waitKey(5) & 0xFF
        # if 's' key pressed save the image
        if key == ord('s'):
            save_number = str(save_number)
            savefileimg = "calib_knn/img_" + save_number + ".png"
            savefileThr = "calib_knn/Thr_" + save_number + ".png"
            # cv2.saveimage("save.png", imgOriginalScene)
            cv2.imwrite(savefileimg, frame)
            cv2.imwrite(savefileThr, img_thresh)
            print("image save !")
            save_number = int(save_number)
            save_number = save_number + 1
        if key == 27:  # if the 'q' key is pressed, stop the loop
            camera.release()  # cleanup the camera and close any open windows
            break

    # For image only
    if not loop:
        img_original_scene = imutils.resize(img_original_scene, width=720)
        cv2.imshow("original", img_original_scene)
        imgGrayscale, img_thresh = pp.preprocess(img_original_scene)
        cv2.imshow("threshold", img_thresh)
        img_original_scene = imutils.transform(img_original_scene)
        img_original_scene, new_license = searching(img_original_scene, loop)
        print(f"license plate read from image = {new_license} \n")
        cv2.waitKey(0)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return


def drawRedRectangleAroundPlate(imgOriginalScene, licPlate):
    p2fRectPoints = cv2.boxPoints(licPlate.rrLocationOfPlateInScene)  # get 4 vertices of rotated rect
    cv2.line(imgOriginalScene, tuple(p2fRectPoints[0]), tuple(p2fRectPoints[1]), SCALAR_RED, 2)  # draw 4 red lines
    cv2.line(imgOriginalScene, tuple(p2fRectPoints[1]), tuple(p2fRectPoints[2]), SCALAR_RED, 2)
    cv2.line(imgOriginalScene, tuple(p2fRectPoints[2]), tuple(p2fRectPoints[3]), SCALAR_RED, 2)
    cv2.line(imgOriginalScene, tuple(p2fRectPoints[3]), tuple(p2fRectPoints[0]), SCALAR_RED, 2)


def writeLicensePlateCharsOnImage(imgOriginalScene, licPlate):
    ptCenterOfTextAreaX = 0  # this will be the center of the area the text will be written to
    ptCenterOfTextAreaY = 0

    ptLowerLeftTextOriginX = 0  # this will be the bottom left of the area that the text will be written to
    ptLowerLeftTextOriginY = 0

    sceneHeight, sceneWidth, sceneNumChannels = imgOriginalScene.shape
    plateHeight, plateWidth, plateNumChannels = licPlate.imgPlate.shape

    intFontFace = cv2.FONT_HERSHEY_SIMPLEX  # choose a plain jane font
    fltFontScale = float(plateHeight) / 30.0  # base font scale on height of plate area
    intFontThickness = int(round(fltFontScale * 1.5))  # base font thickness on font scale

    textSize, baseline = cv2.getTextSize(licPlate.strChars, intFontFace, fltFontScale,
                                         intFontThickness)  # call getTextSize

    # unpack roatated rect into center point, width and height, and angle
    ((intPlateCenterX, intPlateCenterY), (intPlateWidth, intPlateHeight),
     fltCorrectionAngleInDeg) = licPlate.rrLocationOfPlateInScene

    intPlateCenterX = int(intPlateCenterX)  # make sure center is an integer
    intPlateCenterY = int(intPlateCenterY)

    ptCenterOfTextAreaX = int(intPlateCenterX)  # the horizontal location of the text area is the same as the plate

    if intPlateCenterY < (sceneHeight * 0.75):  # if the license plate is in the upper 3/4 of the image
        ptCenterOfTextAreaY = int(round(intPlateCenterY)) + int(
            round(plateHeight * 1.6))  # write the chars in below the plate
    else:  # else if the license plate is in the lower 1/4 of the image
        ptCenterOfTextAreaY = int(round(intPlateCenterY)) - int(
            round(plateHeight * 1.6))  # write the chars in above the plate
    # end if

    textSizeWidth, textSizeHeight = textSize  # unpack text size width and height

    ptLowerLeftTextOriginX = int(
        ptCenterOfTextAreaX - (textSizeWidth / 2))  # calculate the lower left origin of the text area
    ptLowerLeftTextOriginY = int(
        ptCenterOfTextAreaY + (textSizeHeight / 2))  # based on the text area center, width, and height

    # write the text on the image
    cv2.putText(imgOriginalScene, licPlate.strChars, (ptLowerLeftTextOriginX, ptLowerLeftTextOriginY), intFontFace,
                fltFontScale, SCALAR_YELLOW, intFontThickness)


def searching(imgOriginalScene, loop):
    licenses = ""
    if imgOriginalScene is None:  # if image was not read successfully
        print("error: image not read from file \n")  # print error message to std out
        os.system("pause")  # pause so user can see error message
        return
        # end if

    # detect plates
    listOfPossiblePlates = DetectPlates.detectPlatesInScene(imgOriginalScene)
    # detect chars in plates
    listOfPossiblePlates = DetectChars.detectCharsInPlates(listOfPossiblePlates)

    if not loop:
        cv2.imshow("imgOriginalScene", imgOriginalScene)

    if len(listOfPossiblePlates) == 0:
        if not loop:  # if no plates were found
            print("no license plates were detected\n")  # inform user no plates were found
    else:  # else
        # if we get in here list of possible plates has at leat one plate

        # sort the list of possible plates in DESCENDING order (most number of chars to least number of chars)
        listOfPossiblePlates.sort(key=lambda possiblePlate: len(possiblePlate.strChars), reverse=True)
        # suppose the plate with the most recognized chars (the first plate in sorted by string length descending
        # order) is the actual plate
        licPlate = listOfPossiblePlates[0]

        if not loop:
            cv2.imshow("imgPlate", licPlate.imgPlate)  # show crop of plate and threshold of plate
            cv2.imshow("imgThresh", licPlate.imgThresh)

        if len(licPlate.strChars) == 0:  # if no chars were found in the plate
            if not loop:
                print("no characters were detected\n")
                return  # show message
            # end if
        drawRedRectangleAroundPlate(imgOriginalScene, licPlate)
        writeLicensePlateCharsOnImage(imgOriginalScene, licPlate)
        licenses = licPlate.strChars

        if not loop:
            print("license plate read from image = " + licPlate.strChars + "\n")  # write license plate text to std out
            # write license plate text on the image

        if not loop:
            cv2.imshow("imgOriginalScene", imgOriginalScene)  # re-show scene image
            cv2.imwrite("imgOriginalScene.png", imgOriginalScene)

    return imgOriginalScene, licenses


if __name__ == "__main__":
    main()
