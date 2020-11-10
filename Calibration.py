import cv2
import numpy as np

import DetectChars
import Main
import Preprocess as pp
import imutils


def nothing(x):
    pass


def calibration(image):
    WindowName1 = "Calibrating Position of image"
    WindowName2 = "Color Thresholding"
    WindowName3 = "Calibrating for Preprocess"

    # make window
    cv2.namedWindow(WindowName2)
    cv2.namedWindow(WindowName3)
    cv2.namedWindow(WindowName1)

    # Load saved data from calibrated value
    (w, h, rotationx, rotationy, rotationz, panX, panY, stretchX, dist, G_S_F_W, G_S_F_H, A_T_B, A_T_W, T_V, Xtrans,
     Ytrans) = np.loadtxt("calibrated_value.txt")

    # convert from load data to xyzwd
    Xtrans = int(round(Xtrans + 100))
    Ytrans = int(round(Ytrans + 100))
    xValue = int(round(100 - (rotationx * 20000.0)))
    yValue = int(round((rotationy * 20000.0) + 100))
    zValue = int(round(100 - (rotationz * 100)))
    wValue = int(round(100 - ((dist - 1.0) * 200.0)))
    dValue = int(round((stretchX - 1.0) * -200.0 + 100))

    # make Trackbar
    cv2.createTrackbar('Xtrans', WindowName1, Xtrans, 200, nothing)  # for rotation in x axis
    cv2.createTrackbar('Ytrans', WindowName1, Ytrans, 200, nothing)  # for rotation in x axis
    cv2.createTrackbar("Xrot", WindowName1, xValue, 200, nothing)  # for rotation in x axis
    cv2.createTrackbar("Yrot", WindowName1, yValue, 200, nothing)  # for rotation in y axis
    cv2.createTrackbar("Zrot", WindowName1, zValue, 200, nothing)  # for rotation in z axis
    cv2.createTrackbar("ZOOM", WindowName1, wValue, 200, nothing)  # for Zooming the image
    cv2.createTrackbar("Strech", WindowName1, dValue, 200, nothing)  # for strech the image in x axis

    switch = '0 : OFF \n1 : ON'
    cv2.createTrackbar(switch, WindowName3, 0, 1,
                       nothing)  # switch to see the preprocess threshold, for more detail see Preprocess.py
    cv2.createTrackbar('G_S_F_W', WindowName3, int(G_S_F_W), 50, nothing)  # GAUSSIAN_SMOOTH_FILTER_SIZE_WEIGHT
    cv2.createTrackbar('G_S_F_H', WindowName3, int(G_S_F_H), 50, nothing)  # GAUSSIAN_SMOOTH_FILTER_SIZE_HEIGHT
    cv2.createTrackbar('A_T_B', WindowName3, int(A_T_B), 50, nothing)  # ADAPTIVE_THRESH_BLOCK_SIZE
    cv2.createTrackbar('A_T_W', WindowName3, int(A_T_W), 50, nothing)  # ADAPTIVE_THRESH_WEIGHT
    cv2.createTrackbar('T_V', WindowName3, int(T_V), 255, nothing)  # THRESHOLD_VALUE

    cv2.createTrackbar("RGBSwitch", WindowName2, 0, 1, nothing)
    cv2.createTrackbar('Ru', WindowName2, 255, 255, nothing)
    cv2.createTrackbar('Gu', WindowName2, 255, 255, nothing)
    cv2.createTrackbar('Bu', WindowName2, 255, 255, nothing)

    cv2.createTrackbar('Rl', WindowName2, 0, 255, nothing)
    cv2.createTrackbar('Gl', WindowName2, 0, 255, nothing)
    cv2.createTrackbar('Bl', WindowName2, 50, 255, nothing)

    # Allocate destination image
    backGround1 = np.ones((100, 500))
    backGround2 = np.ones((100, 500))
    backGround3 = np.ones((100, 500))
    # Loop for get trackbar pos and process it

    while True:
        # Get position in trackbar for change transform
        Xtrans = cv2.getTrackbarPos('Xtrans', WindowName1)
        Ytrans = cv2.getTrackbarPos('Ytrans', WindowName1)
        X = cv2.getTrackbarPos("Xrot", WindowName1)
        Y = cv2.getTrackbarPos("Yrot", WindowName1)
        Z = cv2.getTrackbarPos("Zrot", WindowName1)
        W = cv2.getTrackbarPos("ZOOM", WindowName1)
        D = cv2.getTrackbarPos("Strech", WindowName1)

        # Get position in trackbar for switch
        S = cv2.getTrackbarPos(switch, WindowName3)  # switch for see the calibration threshold

        # Get the value from tracbar and make it ood and value more than 3 for calibrating threshold
        G_S_F_W = makeood(cv2.getTrackbarPos('G_S_F_W', WindowName3))
        G_S_F_H = makeood(cv2.getTrackbarPos('G_S_F_H', WindowName3))
        A_T_B = makeood(cv2.getTrackbarPos('A_T_B', WindowName3))
        A_T_W = makeood(cv2.getTrackbarPos('A_T_W', WindowName3))
        T_V = float(cv2.getTrackbarPos('T_V', WindowName3))

        RGB = cv2.getTrackbarPos("RGBSwitch", WindowName2)

        Ru = cv2.getTrackbarPos('Ru', WindowName2)
        Gu = cv2.getTrackbarPos('Gu', WindowName2)
        Bu = cv2.getTrackbarPos('Bu', WindowName2)

        Rl = cv2.getTrackbarPos('Rl', WindowName2)
        Gl = cv2.getTrackbarPos('Gl', WindowName2)
        Bl = cv2.getTrackbarPos('Bl', WindowName2)

        lower = np.array([Bl, Gl, Rl], dtype=np.uint8)
        upper = np.array([Bu, Gu, Ru], dtype=np.uint8)

        Xtrans = (Xtrans - 100)
        Ytrans = (Ytrans - 100)
        rotationx = -(X - 100) / 20000.0
        rotationy = (Y - 100) / 20000.0
        rotationz = -(Z - 100) / 100.0
        dist = 1.0 - (W - 100) / 200.0
        stretchX = 1.0 + (D - 100) / -200.0
        w = np.size(image, 1)
        h = np.size(image, 0)
        panX = 0
        panY = 0

        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        blnKNNTrainingSuccessful = DetectChars.loadKNNDataAndTrainKNN()  # attempt KNN training
        if blnKNNTrainingSuccessful == False:  # if KNN training was not successful
            print("\nerror: KNN traning was not successful\n")  # show error message
            return
        imaged = imutils.translate(image, Xtrans, Ytrans)
        # Apply transform
        M = imutils.getTransform(w, h, rotationx, rotationy, rotationz, panX, panY, stretchX, dist)
        imgOriginalScene = cv2.warpPerspective(imaged, M, (w, h), cv2.INTER_CUBIC or cv2.WARP_INVERSE_MAP)

        if (S == 1):
            imgGrayscale = pp.extractValue(imgOriginalScene)
            # imgGrayscale = np.invert(imgGrayscale) # last best use this
            imgMaxContrastGrayscale = pp.maximizeContrast(imgGrayscale)
            imgMaxContrastGrayscale = np.invert(imgMaxContrastGrayscale)
            height, width = imgGrayscale.shape
            imgBlurred = np.zeros((height, width, 1), np.uint8)
            imgBlurred = cv2.GaussianBlur(imgMaxContrastGrayscale, (G_S_F_H, G_S_F_W), 0)
            # imgBlurred = np.invert(imgBlurred)
            imgOriginalScene = cv2.adaptiveThreshold(imgBlurred, T_V, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                                     cv2.THRESH_BINARY_INV, A_T_B, A_T_W)

            # imgThresh = np.invert(imgThresh)
            # cv2.imshow("cobaaa", imgThresh)
        if (RGB == 1):
            imgOriginalScene = cv2.inRange(imgOriginalScene, lower, upper)

        # give definition for each initial on image or windows
        cv2.putText(imgOriginalScene, "Press 's' to save the value", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75,
                    (255, 255, 255), 2, bottomLeftOrigin=False)
        cv2.putText(imgOriginalScene, "Press 'o' to out the value", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.75,
                    (255, 255, 255), 2, bottomLeftOrigin=False)
        cv2.putText(imgOriginalScene, "Press 'c' to check the result", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.75,
                    (255, 255, 255), 2, bottomLeftOrigin=False)
        cv2.putText(imgOriginalScene, "Press 'esc' to close all windows", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.75,
                    (255, 255, 255), 2, bottomLeftOrigin=False)

        cv2.putText(backGround1, "X for rotating the image in x axis", (10, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 0, 0), 1, bottomLeftOrigin=False)
        cv2.putText(backGround1, "Y for rotating the image in y axis", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 0, 0), 1, bottomLeftOrigin=False)
        cv2.putText(backGround1, "Z for rotating the image in z axis", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 0, 0), 1, bottomLeftOrigin=False)
        cv2.putText(backGround1, "ZOOM for Zoom in or Zoom out the image", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 0, 0), 1, bottomLeftOrigin=False)
        cv2.putText(backGround1, "S for streching the image", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1,
                    bottomLeftOrigin=False)

        cv2.putText(backGround2, "R,G,B = Red,Green,Blue", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1,
                    bottomLeftOrigin=False)
        cv2.putText(backGround2, "u,l = Upper and lower", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1,
                    bottomLeftOrigin=False)

        cv2.putText(backGround3, "G_S_F_H = GAUSSIAN_SMOOTH_FILTER_SIZE_HEIGHT", (10, 10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 0, 0), 1, bottomLeftOrigin=False)
        cv2.putText(backGround3, "G_S_F_H = GAUSSIAN_SMOOTH_FILTER_SIZE_WEIGHT", (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 0, 0), 1, bottomLeftOrigin=False)
        cv2.putText(backGround3, "A_T_B = ADAPTIVE_THRESH_BLOCK_SIZE", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 0, 0), 1, bottomLeftOrigin=False)
        cv2.putText(backGround3, "A_T_W = ADAPTIVE_THRESH_WEIGHT", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0),
                    1, bottomLeftOrigin=False)
        cv2.putText(backGround3, "T_V = THRESHOLD_VALUE", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1,
                    bottomLeftOrigin=False)

        # Show in window
        cv2.imshow("image", imgOriginalScene)
        cv2.imshow(WindowName1, backGround1)
        cv2.imshow(WindowName2, backGround2)
        cv2.imshow(WindowName3, backGround3)

        ch = cv2.waitKey(5)

        # chomand switch
        if ch == ord('c'):  # press c to check the result of processing
            Main.searching(imgOriginalScene, True)
            cv2.imshow("check", imgOriginalScene)
            cv2.waitKey(0)
            return

        if S == 1 and ch == ord('p'):  # press c to check the result of processing
            imgOriginalScene = np.invert(imgOriginalScene)
            cv2.imwrite("calib.png", imgOriginalScene)
            cv2.imshow("calib", imgOriginalScene)
            return

        if ch == ord('o'):  # press o to see the value
            print("CAL_VAL =")
            print(w, h, rotationx, rotationy, rotationz, panX, panY, stretchX, dist, G_S_F_W, G_S_F_H, A_T_B, A_T_W,
                  T_V, Xtrans, Ytrans)

        if ch == ord('s'):  # press s to save the value
            CAL_VAL = np.array([[w, h, rotationx, rotationy, rotationz, panX, panY, stretchX, dist, G_S_F_W, G_S_F_H,
                                 A_T_B, A_T_W, T_V, Xtrans, Ytrans]])
            np.savetxt('calibrated_value.txt', CAL_VAL)
            print(w, h, rotationx, rotationy, rotationz, panX, panY, stretchX, dist, G_S_F_W, G_S_F_H, A_T_B, A_T_W,
                  T_V, Xtrans, Ytrans)
            print("Value saved !")

        if ch == 27:  # press esc for exit the calibration
            break

    cv2.destroyAllWindows()
    return


def makeood(value):
    if (value % 2 == 0):
        value = value - 1
    if (value < 3):
        value = 3
    return value
