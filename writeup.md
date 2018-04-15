
# Vehicle detection project

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier RBF SVM classifier
* Optionally, we can also apply a color transform and append binned color features, as well as histograms of color, to our HOG feature vector. 
* Note: for those first two steps we must normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use the trained classifier to search for vehicles in images.
* Run the pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./car_not_car.jpg
[image2]: ./HOG_example.jpg
[image3]: ./windows.jpg
[image5]: ./bboxes_and_heat.jpg
[video1]: ./project_video.mp4

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.

### Histogram of Oriented Gradients (HOG)

#### 1. Features extraction from the training images.

The code for feature extraction is contained in the second code cell of the IPython notebook as `hog_features` and `spatial_bin_feat` functions which is getting called by `get_all_features` function.Additional function `color_hist` is present if we need to extract color histogram feature from image.  

I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][image1]

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example using the `HLS` and `YUV` color space with HOG parameters of `orientations=10`, `pixels_per_cell=(16, 16)` and `cells_per_block=(2, 2)`:


![alt text][image2]

#### 2. Final choice of HOG parameters.

I tried various combinations of parameters and finally i have taken `S channel` of `HLS` and `Y channel` of `YUV` ,as these channels identify images in all kind of lighting conditions.HOG parameter are choosen such that feature vector will be of minimum size. 

#### 3. Training classifier using selected HOG features

I trained a SVM with `rbf` kernel using the generated feature vector for both car and non car images and get accuracy of 98.7 percent.Training time was less than a minute.The code for training SVM is contained in the 8th code cell of the IPython notebook as function `train_model`.

### Sliding Window Search

#### 1. implementation 

I decided to search in possible areas where a vehicle can appear so I have avoided searching in upper half,very near to our own vehicle,extreme left of the image.The code sliding window is present in 14th block of IPython notebook inside function `slide_window` Below image shows the possible areas of windowing:

![alt text][image3]

#### 2. PIpeline and performance optimization

Ultimately I searched on 2 different channel HOG feature in the feature vector, which provided a nice result.To get a good performance I have used feature vector of size 1.1k and windows of different size and density as the nearer object is going to be bigger in shape and distant object appears to be smaller, so bigger and smaller windows are chossen respectively.


### Video Implementation

#### 1. Link to final video output  
Here's a [link to my video result](./project_video.mp4)


#### 2. False positives and combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` as bounding boxes overlaid on the 6 different frames of video:

### Here are six frames and their corresponding heatmaps with  bounding boxes:

![alt text][image5]


### Discussion

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.

* One of the challenge was to choose feature vector as there is a tradeof between feture vector size and training time for model. A feture vector of bigger size may take a long time to train a model which results a bigger model with long prediction time.So I have choosen a feature vector of minimum size.

* As I was doing my feature extraction and model training on low configuration system so I make use pickle to keep minimum code in memory.

* I have tried choosing SVC with linear kernel but it results in low accuracy (less than 95 percent),so I have used rbf kernel for SVC.

* To avoid wobbly windows on cars i have used averaging of heatmap.

* This model is trained on special kind of images for car and non car so it might fail to detect other vehicles or pedestrian on road.

* As we have done windowing on right side of our car so in a scenario when we are driving on middle lane,it wont be available to detect objects/cars on left. To overcome this we need to increase the windowing span.

