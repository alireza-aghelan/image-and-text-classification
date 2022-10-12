# image-and-text-classification
In this project, we train a text classifier for captions and an image classifier for images to classify them.
the dataset we have contains a number of images. for each image, there is a text file with the same name that describes the corresponding image in 5 sentences.
It should be noted that the label of each file is the name of the folder in which it is placed.

Our dataset has a total of 19 categories, and for each category, 28 images are placed in the most folder to train our model with, and 20 images are in the test folder with which we evaluate the performance of our final model.

Examples of images and captions in the dataset
![image](https://user-images.githubusercontent.com/47056654/195458678-3d79dac4-5d3b-4db0-adbf-3db2ac6a4f97.png)

<img width="468" alt="image" src="https://user-images.githubusercontent.com/47056654/195458344-4d328b8e-ee57-4006-b9df-dfb8b53e3eba.png">

To classify the images, we used the pre-trained DenseNet model. DenseNet is one of the new discoveries in neural networks for visual object recognition. DenseNet is quite similar to ResNet with some basic differences.
DenseNet is specifically designed to improve the reduced accuracy caused by vanishing gradients in deep neural networks.

<img width="468" alt="image" src="https://user-images.githubusercontent.com/47056654/195458471-3da407c8-7564-4788-a1fd-ded119ac2e8e.png">

Our model was trained with 20 epochs and achieved a validation accuracy of 91%

We used Scikit Learn Python library for Text Classification
The general steps are as follows:
Text pre-processing - Converting text into numbers - Text Classification - Model evaluation

We did the following steps for text preprocessing: 
remove all the special characters - remove all single characters - remove single characters from the start - substitute multiple spaces with single space - remove prefixed 'b' - convert to lowercase – lemmatization

The next step will be to convert text to numbers, there are different approaches to converting text to the corresponding numerical form.
The bag of Words model and the Word Embedding model are two common methods. In this project, we will use the first model to convert our text into numbers.

We used Random forest algorithm to train our model and finally we reached 76% accuracy.


 
