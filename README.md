# ALPR Indonesia
Automatic license plate recognition for Indonesian plate **(White on black)**<br>

This code was the adjustment version from :<br>
https://github.com/MicrocontrollersAndMore/OpenCV_3_License_Plate_Recognition_Python

## Methods:
Object detection is manualy segmenting plate, and recognize each candidate number or alphabet using knn method.<br>
Here for the detail :
- [Video Explanation](https://www.youtube.com/watch?v=fJcl6Gw1D8k)
- [KNN](https://docs.opencv.org/3.4/d5/d26/tutorial_py_knn_understanding.html)
<br>

**Note** : <br>It is recommended to use a newer method like yolo or ssd. <br>
or you can read the state of the art of object detection <br>
[State-of-the-art of Object Detection ](https://paperswithcode.com/task/object-detection)

## Prerequest
- python==3.6
- opencv-python==4.1.1.26
- numpy==1.17.3<br>
or install using<br>
`pip install -r requirements.txt`

## How to run (Linux):
- Calibrate the camera and threshold<br>
  `python Main.py -c <image_dir>` 
- Cam<br>
  `python Main.py`
- Image<br>
  `python Main.py -i <image_file_dir>`
- Video<br>
  `python Main.py -v <video_file_dir>`

## Retrain
Retrain process will update classifications.txt and flattened_images.txt files<br>
`python GenData.py -d = <train_image>`<br>
example : <br>
`python GenData.py -d = train_image/train2.png`<br>
note: *Just input base on marked object one by one and press esc to exit the training process*

### Check the model
`python TrainAndTestData.py -d = train_image/train2.png`

## Tools
- Invert image:<br>
`python invert_imageData.py -d = train_image/train2.png`
