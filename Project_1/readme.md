## Convolutional Neural Network Perspective on FLAME Dataset

### How to run the code 

####Steps:
#####1. In the Code_files folder run the main.py file  
        -> This will start the program and load the images
#####2. The program will ask user to input " 1 for training , 2 for Loading the saved model for evaluation" 
        -> If the user wants to train a new model then press 1 else if the user wants to load the saved modela nd weights and just wants to evaluate then press 2
        
#####3. The program will print the evaluation matrices. The program uses all the 3 files  
        -> main.py, PreProcessing.py and Xception_model.py


### Dataset
* The dataset is downloaded from IEEE dataport and you can download datasets from [here](https://essexuniversity-my.sharepoint.com/:f:/g/personal/hr17576_essex_ac_uk/EplQh6rwA8pJhHP0jKfg6-kBVHyb1BE9TCAj4MVR0tyOEA?e=Uo6PLD).
* Have uploaded the Data to Kaggle environment [link](https://www.kaggle.com/smrutisanchitadas/flame-dataset-fire-classification)
* Training/Validation dataset: This dataset has 39,375 frames that are resized to 254x254 and the image format is JPG. This data is in a directory called training, which contains 2 sub-directories, Fire and No Fire. There are 25018 images in fire class and 14357 images in No Fire Class
* Test dataset : Test datset has 8,617 frames that are labeled.  which contains 2 sub-directories, Fire and No Fire. There are 5137 images in fire class and 3480 images in No Fire Class

Data Pre Processing

Data Augmentaion is done using image generator. 

### Model
4 Different Models were Trained on FLAME Dataset to get the best model
1. Simple CNN Model : A simple CNN model was built in Assignment 1 to baseline the accuracy and compare with other Trasfer learning models
2. VGG 16 - VGG 16 Model is trained
3. Inception V3 
4. Xception  - We acheived best results with Xception Model.

![BaseModel:Simple CNN](https://github.com/smrutisanchita/CE888/blob/main/Project_1/val_acc.JPG). 

## Requirements
* os
* cv2
* numpy
* Keras 
* Tensorflow
* scikitlearn
* matplotlib.pyplot
* Seaborn
* pathlib
* PIL

## Code
This code is run in Kaggle Kernel with GPU Enabled

## Results
* Below is the graph for accuracy and Loss of all the Models on Validation dataset
![Accuracy](https://github.com/smrutisanchita/CE888/blob/main/Project_1/val_acc.JPG)

![Loss](https://github.com/smrutisanchita/CE888/blob/main/Project_1/val_loss.JPG)


