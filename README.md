# ALPR-Indonesian-plate
This code is for automatic license plate recognition in indonesian plate, which have black background and white number

method we use was k-Nearest Neighbour for recognize chars in image

for more information about knn:
https://docs.opencv.org/3.4/d5/d26/tutorial_py_knn_understanding.html

## for use in linux just:

python Main.py -c directoryFileImage = calibrating the camera and threshold

python Main.py   = use with cam

python Main.py -i directoryFileImage = recognition an image

python Main.py -v directoryFileVideo = recognition a video

## To retrain training data of classifications.txt and flattened_images.txt:

python GenData.py -d = train_image/train2.png

after that you train by yourself, one by one. just input base on marked object
and press esc to exit the training process

## to check your classification is good enough

python TrainAndTestData.py -d = train_image/train2.png

## if you want to invert the image that you want to train
python invert_imageData.py -d = train_image/train2.png
