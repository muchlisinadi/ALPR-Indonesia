# ALPR-Indonesian-plate
Automatic license plate recognition for Indonesian plate (black background and white number)

This code was the adjustment version from :
https://github.com/MicrocontrollersAndMore/OpenCV_3_License_Plate_Recognition_Python

## Methods:
- [How it works](https://www.youtube.com/watch?v=fJcl6Gw1D8k)
- [KNN](https://docs.opencv.org/3.4/d5/d26/tutorial_py_knn_understanding.html)

## How to run (Linux):
- Calibrate the camera and threshold
  `python Main.py -c <image_dir>` 
- Cam
  `python Main.py`
- Image
  `python Main.py -i <image_file_dir>`
- Video
  `python Main.py -v <video_file_dir>`

## Retrain
Retrain process will update classifications.txt and flattened_images.txt files
`python GenData.py -d = <train_image>`
example : `python GenData.py -d = train_image/train2.png`<br>
note: *Just input base on marked object one by one and press esc to exit the training process*

### Check the classification model
`python TrainAndTestData.py -d = train_image/train2.png`

## Tools
### invert image for train process
`python invert_imageData.py -d = train_image/train2.png`
