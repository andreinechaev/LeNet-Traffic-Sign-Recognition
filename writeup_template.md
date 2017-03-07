#**Traffic Sign Recognition** 

##Writeup Template

###You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

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
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

###Data Set Summary & Exploration

####1. Provide a basic summary of the data set and identify where in your code the summary was done. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

The code for this step is contained in the second code cell of the IPython notebook.  

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of test set is 12630
* The shape of a traffic sign image is 32x32x3
* The number of unique classes/labels in the data set is 43

####2. Include an exploratory visualization of the dataset and identify where the code is in your code file.

for the provided data I used only one image.
```import matplotlib.pyplot as plt
import random
import cv2
# Visualizations will be shown in the notebook.
%matplotlib inline

plt.figure(figsize=(8,8))

index = random.randint(0, n_train)
image = X_train[index].squeeze()
plt.figure(figsize=(1,1))
print(y_train[index])
plt.imshow(image)
```

This snippet evolved in a 8x8 grid

```
signs_f = './test_signs/'

imgs = []
exts = []
for f in os.listdir(signs_f):
    if not f.endswith('.jpg'):
        continue
    exts.append(int(f.split('.')[0]))
    img = cv2.imread(signs_f + f)
    img = cv2.resize(img, (32,32))
    normalize(img)
    imgs.append(img)

plt.figure(figsize=(10,10))
index = 0
for img in imgs:
    axis = plt.subplot(4, 4, index+1)
    plt.imshow(img)
    index += 1
plt.show()
```

###Design and Test a Model Architecture

####1. Describe how, and identify where in your code, you preprocessed the image data. What tecniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc.

First, I made all images grayscaled using OpenCV lib. Since our tensor expect an image with shape 32x32x1 and cv2 return a 2d `np array` 32x32, all images was reshaped later to the appropriate matrix. Grayscale images are easier to process by a network, since it will work only with layers with depth equal 1. 
```
def to_grayscale(img):
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    
gray_images = [to_grayscale(img)[:,:, np.newaxis] for img in images]
```

at the moment of loading images, each one was normalized and resized using same OpenCV lib. Normalization is the process of reducing noize. Resizing prepares images for future use in our tensors. 

```
def normalize(img, beta=255):
    cv2.normalize(img, img, alpha=0, beta=beta, norm_type=cv2.NORM_MINMAX)

def image_from_file(path, shape=(32, 32)):
    img = cv2.imread(path)
    img = cv2.resize(img, shape)
    normalize(img)
    return img
```


![alt text][image2]

####2. Describe how, and identify where in your code, you set up training, validation and testing data. How much data was in each set? Explain what techniques were used to split the data into these sets. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, identify where in your code, and provide example images of the additional data)

The code for splitting the data into training and validation sets is contained in the fifth code cell of the IPython notebook.  

To cross validate my model, I randomly split the training data into a training set and validation set. I did this by using `sklearn.model_selection.train_test_split` to giving 20% of data to the validation set.   

Number of training examples = 21312
Number of validation examples = 5328
Number of testing examples = 12569


####3. Describe, and identify where in your code, what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

The code for my final model is located in the seventh cell of the ipython notebook. 

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 Grayscale image   					| 
| Convolution 3x3     	| 1x1 stride, same padding, outputs 32x32x64 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 28x28x6 					|
| Convolution 3x3	    | 1x1 stride, outputs 10x10x16      			|		    |						|												|
| Fully connected		| Input 400 output 120       					|
|						|												|
|						|												|
 


####4. Describe how, and identify where in your code, you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

The code for training the model is located in the tenth cell of the ipython notebook. 

To train the model, I used an the LeNet we learned in the course. I chose 20 epochs because it is enough to show accuracy above 96%. Also, I used CPU for learning the network, so 20 epochs go fast enough. The same reason for the batch size, it's fast enough for such a small data set. Learning rate was gotten experimentally, based on the best accuracy result. 

####5. Describe the approach taken for finding a solution. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* validation set accuracy of 96.4%
* test set accuracy of 83.3%

 

###Test a Model on New Images

####1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web, all of them plotted in 10th cell. 
I had the same problem with all images. 
- Resolution. There supposed to be a better network and perhaps bigger data set to create a network good enough for understanding images with such low resolution
- Artifacts (watermarks). Most of my images contain watermarks that makes it more difficult to determine the sign type. Although, it makes images cloer to reality

Plus, some images have similar features. Particularly numbers. Perhaps reducing strides and increasing the resolution of pictures would improve the situation.

####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. Identify where in your code predictions were made. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

The code for making predictions on my final model is located in the 11th and 12th cells of the Ipython notebook.

83.3 percent of accuracy comparing to the validation accuracy of 96%. 

####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction and identify where in your code softmax probabilities were outputted. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 13th cell of the Ipython notebook.

The prediction level varies but not too much. Some pictures where determine with almost 100% probability.
