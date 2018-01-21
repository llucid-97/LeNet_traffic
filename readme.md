
# **Traffic Sign Recognition** 

## Writeup

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/visualization.png "Visualization"
[image2]: ./examples/distribution.png "Histogram"
[image3]: ./examples/grayscale.jpg "Grayscaling"
[image4]: ./examples/LeNet.png "TensorBoard"
[image5]: ./LeNet/examples/inference/79.jpeg
[image6]: ./LeNet/examples/inference/89.jpeg
[image7]: ./examples/inference/789.jpeg
[image8]: ./examples/inference/867.jpeg
[image9]: ./examples/inference/7888887.jpeg
[image10]: ./examples/inference/images.jpeg
[image11]: ./examples/inference/images6.jpeg
[image12]: ./examples/inference/sign1.jpeg
[image13]: ./examples/topK.png "top K"




## [Rubric Points](https://review.udacity.com/#!/rubrics/481/view)
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/ihexx/LeNet_traffic/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas library to calculate summary statistics of the traffic
signs data set:

Number of training examples = 34799
Number of Validation examples = 4410
Number of testing examples = 12630
Image data shape = (34799, 32, 32, 3)
Number of classes = 42


#### 2. Include an exploratory visualization of the dataset.

This is what an example image looks like:

![alt text][image1]

Here is an exploratory visualization of the data set.
It shows the distribution of how frequently classes occur in the dataset.

![alt text][image2]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. 

As a first step, I decided to convert the images to grayscale because every class can be successfully classified by a human in greyscale.
The task can be done in greyscale, so including colour channels is just adding needless complexity to the dataset which would be harder for the model to parse

Here is an example of a traffic sign image before and after grayscaling.

![alt text][image3]

As a last step, I normalized the image data because it is easier for the optimizer to find the path of steepest descent when components of features have similar variance.
I used **Batch Normalization** on input data instead of "whitening" (mean shift divided by range). Batch Normalization approximates the input normalization based on only the batch present. I guessed that since the dataset is shuffled each epoch, each time an image is normalized, the approximation is slightly different.
This "noise" in normalization approximation augmented the dataset slightly, and gave marginally improved accuracy over "whitening".


I decided **not** to generate additional data because I had already reached the minimum required validation accuracy.

#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model architecture consisted:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| RGB2GRAY         		| 32x32 Greyscale image   							| 
| Batch Normalization   | 32x32 Greyscale image   							| 
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x6 	|
| RELU					|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6 				|
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 10x10x16 	|
| RELU							|		
| Max pooling	      	| 2x2 stride,  outputs 5x5x16 |										|
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 1x1x400 	|
| RELU					|
| Flatten	|  400 Units	|
| Dense (Fully Connected)				| 43 ([1x400]*[400x43] MatMul	|
| Dropout Regularization						|	50% Keep Probability during training		|
| Softmax						|			|

I used keras for building and training as I was behind the deadline and needed to finish quickly

#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used an RMSProp optimizer with the following hyper parameters:
| Parameter         		|     Value	        					| 
|:---------------------:|:---------------------------------------------:| 
| Learning rate         		| 0.001   							| 
| Rho         		| 0.9   							|
| Decay         		| 0  							|
I ran for 50 Epochs with a batch size of 100 

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 99.6%
* validation set accuracy of 95.3%
* test set accuracy of 94.6%

I initially went for 1 fully connected layer on the source image while setting up my pipeline

It was underfitting.
I kept adding layers and retraining: exploring the solution space using the [Sermanet/LeCun model](http://yann.lecun.com/exdb/publis/pdf/sermanet-ijcnn-11.pdf) as a guide.

I kept increasing complexity until it began to overfit.
I tried several different combinations of filter size, model depth, activation functions, in-model normalizations, all to see how the model performed differently.
The goal here was to play with architectures until I found something that overfit only a little, then I would regularize that.
I used dropout. It was the only one I tried, and it gave me an extra 4% i needed to beat the minimum requirement.

The dataset is small. Dense models are a lot more susceptible to overfit, which is why I went for convolutional models. I am yet to try it on layers within the model. I think it would be more effective there than on the final layer

The final model has close accuracies on the training and validation sets. It is enough to meet the spec, but it is still overfitting.
Still, I didn't want to throttle its representational power just yet: I felt data augmentation would be the next logical step before further tuning, but that would be a job for the next traffic sign classifier. 

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

 ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8] ![alt text][image9]

| Image			        |     Difficulty	        					| 
|:---------------------:|:---------------------------------------------:| 
| Speed Limit 120     		| Perspective Skew   (not in dataset)									| 
| Children Crossing     		| Tree in sharp focus "distracts" lower-level activations   									| 
| Wild Animals Crossing     		| Perspective Skew   									| 
| Speed Limit 130     		| Number. Lots of speed limit classes are similar general shape, but with different numbers   									| 
| Children Crossing  (2)  		| 						 n/a |

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set

Here are the results of the prediction:
| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Speed Limit 120     		| No Entry			| 
| Children Crossing     		| Children Crossing   									| 
| Wild Animals Crossing     		| Wild Animals Crossing| 
| Speed Limit 130     		| Speed Limit 60| 
| Children Crossing  (2)  		| 						 Roundabout mandatory |


The model was able to correctly guess 3 of the 5 traffic signs, which gives an accuracy of 60%. This compares unfavorably to the accuracy on the test set of 94%

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability.

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.
![alt text][image13]
