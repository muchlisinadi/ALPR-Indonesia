
ap = argparse.ArgumentParser()
#ap.add_argument("-f", "--face", required = True,
    #help = "path to where the face cascade resides")
ap.add_argument("-v", "--video",
    help = "path to the (optional) video file")
args = vars(ap.parse_args())

# construct the face detector
#fd = FaceDetector(args["face"])

# if a video path was not supplied, grab the reference
# to the gray
if not args.get("video", False):
    camera = cv2.VideoCapture(0)

# otherwise, load the video
else:
    camera = cv2.VideoCapture(args["video"])

# keep looping
while True:
    # grab the current frame
    (grabbed, frame) = camera.read()

    # if we are viewing a video and we did not grab a
    # frame, then we have reached the end of the video
    if args.get("video") and not grabbed:
        break

    # resize the frame and convert it to grayscale
    frame = imutils.resize(frame, width = 300)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # detect faces in the image and then clone the frame
    # so that we can draw on it
    #faceRects = fd.detect(gray, scaleFactor = 1.1, minNeighbors = 5,
       # minSize = (30, 30))
    frameClone = frame.copy()
    frameClone2 = frame.copy()
    [ imgGrayscale, imgThresh ] = pp.preprocess(frameClone)
    hasil = outhasil (frameClone2)
    #imgOriginalScenes = main(frameClone)


    # loop over the face bounding boxes and draw them
    #for (fX, fY, fW, fH) in faceRects:
       # cv2.rectangle(frameClone, (fX, fY), (fX + fW, fY + fH), (0, 255, 0), 2)

    # show our detected faces
    cv2.imshow("Face", frameClone)
    cv2.imshow("grey",gray)
    cv2.imshow("threshold", imgThresh)
    cv2.imshow("hasil", hasil)

    # if the 'q' key is pressed, stop the loop
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# cleanup the camera and close any open windows
camera.release()
cv2.destroyAllWindows()


###################################################################################################
def outhasil( frameClone ):

    imgOriginalScene  = frameClone                # open image

    if imgOriginalScene is None:                            # if image was not read successfully
        print("\nerror: image not read from file \n\n")      # print error message to std out
        os.system("pause")                                  # pause so user can see error message
        return                                              # and exit program
    # end if
    listOfPossiblePlates = DetectPlates.detectPlatesInScene(imgOriginalScene)           # detect plates

    listOfPossiblePlates = DetectChars.detectCharsInPlates(listOfPossiblePlates)        # detect chars in plates

    cv2.imshow("imgOriginalScene", imgOriginalScene)            # show scene image

    if len(listOfPossiblePlates) == 0:                          # if no plates were found
        print("\nno license plates were detected\n")             # inform user no plates were found
    else:                                                       # else
                # if we get in here list of possible plates has at leat one plate

                # sort the list of possible plates in DESCENDING order (most number of chars to least number of chars)
        listOfPossiblePlates.sort(key = lambda possiblePlate: len(possiblePlate.strChars), reverse = True)

                # suppose the plate with the most recognized chars (the first plate in sorted by string length descending order) is the actual plate
        licPlate = listOfPossiblePlates[0]

        cv2.imshow("imgPlate", licPlate.imgPlate)           # show crop of plate and threshold of plate
        #cv2.imshow("imgThresh", licPlate.imgThresh)

        if len(licPlate.strChars) == 0:                     # if no chars were found in the plate
            print("\nno characters were detected\n\n")       # show message
            return                                          # and exit program
        # end if

        drawRedRectangleAroundPlate(imgOriginalScene, licPlate)             # draw red rectangle around plate

        print("\nlicense plate read from image = " + licPlate.strChars + "\n")       # write license plate text to std out
        print("----------------------------------------")

        writeLicensePlateCharsOnImage(imgOriginalScene, licPlate)           # write license plate text on the image

        #cv2.imshow("imgOriginalScene", imgOriginalScene)                # re-show scene image

        #cv2.imwrite("imgOriginalScene.png", imgOriginalScene)           # write image out to file

    # end if else
    return imgOriginalScene              # hold windows open until user presses a key

###################################################################################################


# grab the current frame
        (grabbed, frame) = camera.read()

        # if we are viewing a video and we did not grab a
        # frame, then we have reached the end of the video
        if args.get("video") and not grabbed:
            break

        # resize the frame and convert it to grayscale
        frameClone = imutils.resize(frame, width = 600)
        #cv2.imshow("clone", frameClone)

        imgOriginalScene  = frameClone

        if imgOriginalScene is None:                            # if image was not read successfully
            print("error: image not read from file \n")      # print error message to std out
        # end if

        listOfPossiblePlates = DetectPlates.detectPlatesInScene(imgOriginalScene)           # detect plates

        listOfPossiblePlates = DetectChars.detectCharsInPlates(listOfPossiblePlates)        # detect chars in plates

        if len(listOfPossiblePlates) == 0:                          # if no plates were found
            print("no license plates were detected\n")             # inform user no plates were found
        else:                                                       # else
                    # if we get in here list of possible plates has at leat one plate

                    # sort the list of possible plates in DESCENDING order (most number of chars to least number of chars)
            listOfPossiblePlates.sort(key = lambda possiblePlate: len(possiblePlate.strChars), reverse = True)

                    # suppose the plate with the most recognized chars (the first plate in sorted by string length descending order) is the actual plate
            licPlate = listOfPossiblePlates[0]

            if len(licPlate.strChars) == 0:                     # if no chars were found in the plate
                print("no characters were detected\n")       # show message
            # end if

            drawRedRectangleAroundPlate(imgOriginalScene, licPlate)             # draw red rectangle around plate

            print("license plate read from image = " + licPlate.strChars + "\n")       # write license plate text to std out
            #print("----------------------------------------")

            writeLicensePlateCharsOnImage(imgOriginalScene, licPlate)           # write license plate text on the image

            cv2.imshow("imgOriginalScene", imgOriginalScene)  
            #apa = False              # re-show scene image
            
            #cv2.imwrite("imgOriginalScene.png", imgOriginalScene)           # write image out to file
            
        # end if else 
        cv2.waitKey(100)
        #if cv2.waitKey(1) & 0xFF == ord("q"):
            #break

    # cleanup the camera and close any open windows
    camera.release()
    cv2.destroyAllWindows()


    ss >> w;
    950.000000
    ss >> h;
    712.000000
    ss >> rotationx;
    -0.000150
    ss >> rotationy;
    0.001250
    ss >> rotationz;
    -0.060000
    ss >> stretchX;
    1.095000
    ss >> dist;
    0.550000
    ss >> panX;
    0.000000
    ss >> panY;
    0.000000


setTransform(w, h, rotationx, rotationy, rotationz, panX, panY, stretchX, dist)
    
prewarp = planar,,,,,,,,,


cv::Mat PreWarp::getTransform(float w, float h, 
          float rotationx, float rotationy, float rotationz, 
          float panX, float panY, float stretchX, float dist) {

    float alpha = rotationx;
    float beta = rotationy;
    float gamma = rotationz;
    float f = 1.0;

    // Projection 2D -> 3D matrix
    Mat A1 = (Mat_<double>(4,3) <<
        1, 0, -w/2,
        0, 1, -h/2,
        0, 0,    0,
        0, 0,    1);
    
    // Camera Intrisecs matrix 3D -> 2D
    Mat A2 = (Mat_<double>(3,4) <<
        f, 0, w/2, 0,
        0, f, h/2, 0,
        0, 0,   1, 0);

    // Rotation matrices around the X axis
    Mat Rx = (Mat_<double>(4, 4) <<
        1,          0,           0, 0,
        0, cos(alpha), -sin(alpha), 0,
        0, sin(alpha),  cos(alpha), 0,
        0,          0,           0, 1);

    // Rotation matrices around the Y axis
    Mat Ry = (Mat_<double>(4, 4) <<
        cos(beta), 0, sin(beta), 0,
        0, 1, 0, 0,
        -sin(beta), 0,  cos(beta), 0,
        0,          0,           0, 1);

    // Rotation matrices around the Z axis
    Mat Rz = (Mat_<double>(4, 4) <<
        cos(gamma), -sin(gamma), 0, 0,
        sin(gamma), cos(gamma), 0, 0,
       0, 0, 1, 0,
        0,          0,           0, 1);

    Mat R = Rx*Ry*Rz;

    // Translation matrix on the Z axis 
    Mat T = (Mat_<double>(4, 4) <<
        stretchX, 0, 0, panX,
        0, 1, 0, panY,
        0, 0, 1, dist,
        0, 0, 0, 1);


    return A2 * (T * (R * A1));

  }

  cv2.warpPerspective(image, warped_image, transform, image.size(), INTER_CUBIC | WARP_INVERSE_MAP)

  rows,cols,ch = imgOriginalScene.shape
        pts1 = np.float32([[50,50],[200,50],[50,200]])
        pts2 = np.float32([[10,100],[200,50],[100,250]])
        M = cv2.getAffineTransform(pts1,pts2)
        imgOriginalScene = cv2.warpAffine(imgOriginalScene,M,(cols,rows))


def getTransform (w, h, rotationx, rotationy, rotationz, panX, panY, stretchX, dist):
    alpha = rotationx;
    beta = rotationy;
    gamma = rotationz;
    f = 1.0;

    A1 = np.matrix([[1, 0, -w/2], [0, 1, -h/2],[0, 0,0],[0, 0,1]])
    print(A1)
    A2 = np.matrix([[f, 0, w/2, 0], [0, f, h/2, 0],[0, 0, 1, 0]])
    print(A2)
    Rx = np.matrix([[1, 0, 0, 0],[0, math.cos(alpha), -(math.sin(alpha)), 0],[0, math.sin(alpha),  math.cos(alpha), 0],[0, 0, 0, 1]])
    print(Rx)
    Ry = np.matrix([[math.cos(beta), 0, math.sin(beta), 0],[0, 1, 0, 0],[-(math.sin(beta)), 0,  math.cos(beta), 0],[0, 0, 0, 1]])
    print(Ry)
    Rz = np.matrix([[math.cos(gamma), -(math.sin(gamma)), 0, 0],[math.sin(gamma), math.cos(gamma), 0, 0],[0, 0, 1, 0],[0, 0, 0, 1]])
    print(Rz)
    R = Rx*Ry*Rz
    print(R)
    T = np.matrix([[stretchX, 0, 0, panX],[0, 1, 0, panY],[0, 0, 1, dist],[0, 0, 0, 1]])
    print(T)
    M = A2*(T*(R*A1))
    print(M)
    return M
    