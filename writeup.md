# Vehicle Detection

#### Vehicle Detection Project

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector.
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./output_images/boxes_hog_vs_standard.png
[image2]: ./output_images/color_spaces.png
[image3]: ./output_images/feature_normalization.png
[image4]: ./output_images/heatmap_label.png
[image5]: ./output_images/sliding_windows.png
[image6]: ./output_images/visualization_hog.png
[image7]: ./output_images/visualization_spatial_colorhist.png
[image8]: ./output_images/heatmap_label2.png
[image9]: ./output_images/heatmap_label3.png
[image10]: ./output_images/heatmap_label4.png
[image11]: ./output_images/heatmap_label5.png
[image12]: ./output_images/heatmap_label6.png
[image13]: ./output_images/detection_filter.gif

#### Submission

My project includes the following files:
* `vehicle_detection.ipynb` containing the pipeline
* `classifier_data.p` classifier used for the final video
* `project_video_output.mp4` the final output video
* `writeup.md` summarizing the results


## Input data

I used GTI and KITTI datasets for [vehicles](https://s3.amazonaws.com/udacity-sdc/Vehicle_Tracking/vehicles.zip) and [non-vehicles](https://s3.amazonaws.com/udacity-sdc/Vehicle_Tracking/non-vehicles.zip) images. As written in lecture the GTI car folders are time-series data containing nearly identical images. To optimize classifier I sorted and filtered every 8th image, as it seems that similar cars happen to be within groups of eight.  After random shuffling and labeling (2nd code cell) I have `6320` cars and `8968` notcars in my dataset. Splitting them results in a dataset size of `12230`/`3058`training/testing images.

This is a visualize data in different color spaces (3rd code cell):

![alt text][image2]

## Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained in the 4th code cell of the IPython notebook. The code for calculating features is mainly taken from the lectures. The 5th code cell has the code for visualizing the features including spatial binning, color histograms and HOG.

I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like along with spatial binning and color histograms. Here is an example using the `YUV` color space and HOG parameters of `orientations=9`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:

![alt text][image6]

I also used spatial binning and color histograms as features.

![alt text][image7]

#### 2. Explain how you settled on your final choice of HOG parameters.

I tried various combinations of parameters and evaluated their performance with linear SVC, details can be found later. Visual inspection show that setting `pixels_per_cell` to `8` or `10` produces identifiable car outlines. At `10` feature calculation time decreases by 32%, but the edges are missing in 64x64 images. Increasing it further produces much worse result visually. Decreasing HOG `orientations` to `8` does not look as good as `9`, finer resolution of histogram is not required. As I saw later HOG performs much better on histogram equalized grayscale than raw grayscale. This is where `cells_per_block` comes into play, I had better results when block normalization covered more area.

#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I defined a method for training linear SVC in the 9th code cell. This was used extensively for evaluating different feature performance, details can be found later in the writeup. The `train_svc` function uses the whole dataset, calculated features are normalized as the image below shows. During feature evaluation default SVC parameters are used, later `C = 1` is set.

Visualization of feature scaling (11th code cell):

![alt text][image3]

## Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

Helper functions reside in 12th code cell. Function `draw_boxes` can draw boxes optionally with random colors. Function `generate_windows` is used for sliding window, it places windows with overlap at horizontally centered positions. Function `search_windows` extracts features for windows and makes predictions returning 'hot windows'. Function `find_cars` is for the same purpose, but it performs HOG subsampling hoping that it fastens car detection, unfortunately it is not the case for my window configuration. Code cells 13 and 14 demos standard feature extraction and HOG subsampling on test images. The same methods were tested on video as well. Standard way gives 2.8 it/s, while HOG subsampling gives 2.2 it/s for this window setup. Quite surprising. I used function `search_windows` afterwards.

This is the window setup that I came up with. Visually this seem to cover most of the search area. Windows have different sizes, namely `48`, `96`, `128` and `256` (or in scaling 0.75x, 1.5x, 2x and 4x). This setup should detect all the different car sizes at different distances. It consists of 145 windows. Unfortunately it needs to cover the area where cars appear in opposite way coming closer. The trained SVC also fired on many of these cars.

![alt text][image5]

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

I explored different color spaces and different feature parameters to see which settings perform best with linear SVM classifier. I tested spatial binning, color histograms and HOG features separately. The tests were performed using linear SVC with default parameters.

* Results for only spatial binning

|Color Space |Spatial Size | Feature Vector Length | Test Accuracy|
|:--:|:--:|:-:|:--:|
|RGB|(32, 32)|3072|0.939|
|RGB|(16, 16)|768|0.9444|
|RGB|(12, 12)|432|0.9441|
|RGB|(10, 10)|300|0.934|
|RGB|(8, 8)|192|0.929|
|HSV|(12, 12)|432|0.919|
|YUV|(12, 12)|432|0.947|*
|YUV|(16, 16)|768|0.942|
|GRAY|(20, 20)|400|0.893|
|GRAY_HISTEQ|(20, 20)|400|0.878|


* Results for only color histograms

|Color Space |Histogram bins | Feature Vector Length | Test Accuracy|
|:--:|:--:|:-:|:--:|
|RGB|48|144|0.961|
|RGB|32|96|0.959|
|RGB|24|72|0.956|
|RGB|16|48|0.939|
|RGB|12|36|0.917|
|HSV|24|72|0.968|
|YUV|24|72|0.969|
|YUV|32|96|0.971|
|YUV|40|120|0.971|
|YUV|48|144|0.972|*
|YUV|52|144|0.972|
|YUV|64|192|0.970|


* Results for only HOG features

|Color Space | Channel | Orientations | Pixels per cell | Cells per block | Feature VLength | Test Accuracy|
| :--:| :-:| :--: | :--:| :-:| :--: | :--:|
|RGB|ALL|9|8|2|5292|0.946|
|HSV|ALL|9|8|2|5292|0.955|
|YUV|ALL|9|8|2|5292|0.973|
|GRAY|-|9|8|2|1764|0.938|
|GRAY_HISTEQ|-|9|8|2|1764|0.946|
|YUV|ALL|9|10|2|2700|0.973|
|YUV|0|9|10|2|900|0.943|
|YUV|1|9|10|2|900|0.958|
|YUV|2|9|10|2|900|0.948|
|YUV|UV|9|10|2|1800|0.966|
|YUV|ALL|9|8|1|1728|0.966|
|YUV|ALL|9|8|4|10800|0.966|
|YUV|ALL|9|8|6|8748|0.968|
|YUV|ALL|9|8|8|1728|0.978|*

I settled on using the marked parameters. Combining spatial binning, color histrograms and HOG features I get accuracy around `0.99`.

With the right feature parameters identified I trained the final linear SVC with `C = 0.01` to achieve a bit more generalization hopefully.

Results of the pipeline for test images are shown here. Yellow boxes are hot windows where SVC predicted cars. Image in the middle is the thresholded heatmap. On the right we can see the final results showing boxes for identifed cars.

![alt text][image4]
![alt text][image8]
![alt text][image9]
![alt text][image10]
![alt text][image11]
![alt text][image12]

---

## Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./project_video_output.mp4)


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

The implemented filter class is `Detector` in 12th code cell. I recorded the positions of positive detections in each frame of the video. These hot windows were accumulated and saved for 20 frames. From these positive detections I created a heatmap and then thresholded that map to identify vehicle positions. Thresholding was performed using a `max` function of a static value and a dynamic value depending on number of accumulated hot windows. The final heatmap is saved and slightly added to next iteration. I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.

This is a demonstration of false positive filtering.

![alt text][image13]

---

## Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

False positive filtering is quite a challenge. Most of my false predictions are cars coming from the opposite direction. Quite understandable. However it should be filtered in this case. There are scenario where much need to be filtered, and another where the single hot window on the image is the real thing.

Largely different level of horizon could make the pipeline fail, just as any other object that was not included in the dataset. City environment might also be a problem.

As much as I would have liked to make my pipeline faster, I could not make it so with HOG subsampling. The next way to approach would be to divide windows to subsets and predict using a different subset each frame.

