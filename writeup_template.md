# **Traffic Sign Recognition** 

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/visualization.jpg "Visualization"
[image2]: ./examples/grayscale.jpg "Grayscaling"
[image3]: ./examples/random_noise.jpg "Random Noise"
[image4]: ./examples/placeholder.png "Traffic Sign 1"
[image5]: ./examples/placeholder.png "Traffic Sign 2"
[image6]: ./examples/placeholder.png "Traffic Sign 3"
[image7]: ./examples/placeholder.png "Traffic Sign 4"
[image8]: ./examples/placeholder.png "Traffic Sign 5"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
You're reading it! and here is a link to my [project code](https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration
#### 1. Summary
I used the numpy library to calculate summary statistics of the traffic signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

#### 2. Exploration

Here is an exploratory visualization of the data set. It is a bar chart showing how the data is distributed, with x-axis being sign label numbers and y-axis being counts. I also plot 43 images from different classes with its classId as title on top.

![Exploration Chart][exploration_chart.png]

### Design and Test a Model Architecture

####1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a first step, I decided to convert the images to grayscale because it results better training speed and based on my experiments, it has a slightly better accuracy. 

Here is an example of a traffic sign image before and after grayscaling.

![Sign Image Before Grayscaling][sign_image_before_grayscaling]
![Sign Image After Grayscaling][sign_image_after_grayscaling]

As a last step, I normalized the image data to the range of [0.1, 0.9] because it converges quicker in my experiments than (pixel - 128)/float(128), which is the range of [-1, 1]. Here is an example of a grayscaled traffic sign image and its normalized image.

![Sign Image After Grayscaling][sign_image_after_grayscaling]
![Sign Image After Grayscaling][sign_image_after_normalized]


#### 2. Model

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 RGB image   							| 
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x16 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride, valid padding, outputs 14x14x16 				|
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 10x10x32 	|
| RELU					|												|
| Dropout	      	| 0.75 keep probability, outputs 10x10x32 				|
| Convolution 3x3     	| 1x1 stride, valid padding, outputs 8x8x64 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride, valid padding, outputs 4x4x64 				|
| Flatten 	    | outputs 1024      									|
| Fully connected		| outputs 200        									|
| RELU					|												|
| Fully connected		| outputs 100        									|
| RELU					|												|
| Fully connected		| outputs 43        									|
| RELU					|												|
|						|												|
|						|												|
 


#### 3. Model training

To train the model, I used an AdamOptimizer with learning rate 0.001 and minimizing the loss by softmax and cross entropy with the logits from the model above. I set batch size of 128 and epochs to 10 for the training.

#### 4. My approach

My final model results were:
* training set accuracy of 0.998
* validation set accuracy of 0.950
* test set accuracy of 0.930

I first tried with the model from Udacity CNN course, 2 conv layers followed by 3 full connected layers. With that i can only get to slight more than 80% accuracy. So I tried to turn learning rate, add more conv layers and the depth of conv layers. From my experiments, I received a good start accuracy rate with learning rate of 0.05, but it was not stable towards the end. Adding one more conv layer and increasing the depth of conv layers helps to improve accuracy. I eventually settled with 3 conv layers and reasonable depths of conv layers, as there was not much improvement in accuracy by depth increase but it took longer to train. With these configuration, I was able to get to more than 90% accuracy with 10 epochs. I then played with image grayscale and adjusted image pixel normalization from the range of [-1, 1], (x-128)/128, to the range [0.1, 0.9], (0.9 - 0.1)*images/255+0.1. With this, I was finally able to reach around 95% accuracy on validation dataset. Beside, I added a dropout activation to the third conv layer to prevent overfitting and from the final accuracy results on three dataset above, the model is ok.
 

### Test the Model on New Images

#### 1. New Images

Here are six German traffic signs that I found on the web:

![Stop sign][sample_images/pic1] ![No entry][sample_images/pic2] ![General caution][sample_images/pic3] 
![Slippery road][sample_images/pic4] ![Speed limit (70km/h)][sample_images/pic5] ![Ahead only][sample_images/pic6]

The first, fourth and fifth images are difficult, as the first one is quite dark, the fourth being a complicated drawing and the fifth of Speed limit 70km/h not being clearly depicted. And the rest three images are relatively easy to identify.

#### 2. The model's predictions on these new traffic signs

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Stop sign      		| 14 Stop sign   									| 
| No entry     			| 17 No entry 										|
| General caution					| 18 General caution											|
| Slippery road	      		| 23 Slippery road					 				|
| Speed limit (70km/h)			| 4 Speed limit (70km/h)      							|
| Ahead only	      		| 35 Ahead only					 				|


The model was able to correctly guess 6 of the 6 traffic signs, which gives an accuracy of 100%. This compares favorably to the accuracy on the test set of 0.930.

#### 3. The model's certain on new images' prediction

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook. The top five softmax for the six sample images are as follows.

The first stop sign:

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 9.99964952e-01         			| 14 Stop   									| 
| 2.22187809e-05     				| 1 Speed limit (30km/h) 										|
| 3.88751414e-06					| 38 Keep right											|
| 2.97820020e-06	      			| 33 Turn right ahead					 				|
| 2.24264568e-06				    | 4 Speed limit (70km/h)      							|

The second no entry sign:

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.00000000e+00         			| 17 No entry   									| 
| 5.42338563e-10     				| 33 Turn right ahead 										|
| 9.21653240e-11					| 14 Stop											|
| 6.56020377e-11	      			| 34 Turn left ahead					 				|
| 1.66341559e-11				    | 23 Slippery Road      							|

The third general caution sign:

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.00000000e+00         			| 18 General caution   									| 
| 3.71913278e-10     				| 27 Pedestrians 										|
| 3.64076885e-14					| 26 Traffic signals											|
| 5.13204441e-18	      			| 11 Right-of-way at the next intersection					 				|
| 2.18185406e-21				    | 33 Turn right ahead      							|

The fourth slippery road sign:

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.00000000e+00         			| 23 Slippery road   									| 
| 8.36287928e-10     				| 28 Children crossing 										|
| 5.46616141e-10					| 19 Dangerous curve to the left											|
| 3.24310262e-12	      			| 20 Dangerous curve to the right					 				|
| 2.63477868e-12				    | 29 Bicycles crossing      							|

The fifth speed limit (70km/h) sign:

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 9.99964237e-01         			| 4 Speed limit (70km/h)   									| 
| 2.84179605e-05     				| 1 Speed limit (30km/h) 										|
| 5.97700910e-06					| 14 Stop											|
| 1.25307747e-06	      			| 0 Speed limit (20km/h)					 				|
| 1.55657432e-07				    | 8 Speed limit (120km/h)      							| 

The sixth ahead only sign:

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 9.99994993e-01,            			| 35 Ahead only   									| 
| 4.98195868e-06     				| 33 Turn right ahead 										|
| 2.92078433e-08					| 34 Turn left ahead											|
| 1.81852347e-08	      			| 9 No passing					 				|
| 1.42763676e-08				    | 38 Keep right      							| 

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
####1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?


