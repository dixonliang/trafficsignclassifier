# **Traffic Sign Recognition** 

## Writeup


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

[image1]: ./histogram.png "histogram"
[image2]: ./60km.png "60km"
[image3]: ./stop.jpg "stop"
[image4]: ./nopassing.png "passing"
[image5]: ./yield.png "yield"
[image6]: ./roadowork.png "roadwork"


Here is a link to my [project code](https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34,799.
* The size of the validation set is 4,410. 
* The size of test set is 12,630. 
* The shape of a traffic sign image is 32x32x3. 
* The number of unique classes/labels in the data set is 43.

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing how the data is distributed by using a histogram. Based on the color of the bar (Blue = Train, Orange = Test, Green = Valid)  we can see for each set, the distrbution between the sign types is the same as well as roughly how many instances of each sign are in each set. 

![alt text][image1]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a first step, I decided to normalize the images because we want to well condition the data so that the mean and variance are standarized. This way, the optimizer can proceed more efficiently. Then, I convereted each image to grayscale for a similar reason as the colors in this case are not relevant in object idenfication. After converting the images to grayscale, our input shape is now 32x32x1. 


#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

I used the standard LeNet model with the additional of a dropout layer. My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 RGB image   							| 
| Convolution 3x3     	| 1x1 stride, 6 zero padding, outputs 28x28x6 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6  				|
| Convolution 3x3	    | 1x1 stride, 16 zero padding, outputs 10x10x6	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 5x5x16   				|
| Dropout       		| Testing = 0.6         						|
| Flatten       		| outputs 400            						|
| Fully connected		| 120 zero padding, ouputs 120					|
| RELU					|												|
| Fully connected		| 84 zero padding, outputs 84 					|
| RELU					|												|
| Fully connected		| 43 zero padding, outputs 43 					|
| Softmax				| Take top probability for prediction  			|
|						|												|
|						|												|
 


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used an epochs of 15 and batch size of 128. The optimizer I used was the Tensor Flow provided Adam Optimizer. The learning rate of my model was 0.001. 

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of ~0.99
* validation set accuracy of >0.93.  
* test set accuracy of ~0.98

The first architecture that was used was the default LeNet model with a learning rate of 0.001, epochs of 10, and batch size 128. Initially, the first few runs, I was getting a validation set accuracy of ~0.89. The first thing I decided to do was to add a dropout layer just before the fully connected run as I noticed the training and training sets were higher than the validaiton set accuracy which implied some over-fitting on the training data. After deciding to do this initially with a keep probability of 0.5, I noticed that I was actually getting worse results if the probability was too low. With a keep probability too high, there wasn't much difference from not having the dropout layer at all. After testing several values, I noticed that 0.6 seemed to improve the model by 1-2%. 

Afterwards, I decided to look at the parameters that were used for the final improvement. The first thing I looked at was the number of epochs to see if any more improvement could be given if we just went through the networks more times to train it more. Adding 5 more pass throughs seemed to improve the model by another 1-2% which passed me through the threshold of 0.93 averaging anywhere between ~0.93 and ~0.95 validation accuracy. 


### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image2] ![alt text][image3] ![alt text][image4] 
![alt text][image5] ![alt text][image6]

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Stop Sign      		| Stop sign   									| 
| U-turn     			| U-turn 										|
| Yield					| Yield											|
| 100 km/h	      		| Bumpy Road					 				|
| Slippery Road			| Slippery Road      							|


The model was able to correctly guess 3 of the 5 traffic signs, which gives an accuracy of 60%. This compares unfavorably to the data set that was provided which was getting accuracy of >0.93 on the validation set. 


#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For the first image, the model is relatively sure that this is a stop sign (probability of 0.6), and the image does contain a stop sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .60         			| Stop sign   									| 
| .20     				| U-turn 										|
| .05					| Yield											|
| .04	      			| Bumpy Road					 				|
| .01				    | Slippery Road      							|


For the second image ... 


