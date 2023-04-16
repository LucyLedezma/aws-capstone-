**NOTE:** This file is a template that you can use to create the README for your project. The **TODO** comments below will highlight the information you should be sure to include.

# Inventory Monitoring at Distribution Centers

Distribution centers often use robots to move objects. Objects are carried in bins which can contain multiple of them.
So, I am going to build a model that can count the number of objects in each bin.

## Project Set Up and Installation
**OPTIONAL:** If your project has any special installation steps, this is where you should put it. To turn this project into a professional portfolio project, you are encouraged to make your `README` detailed and self-explanatory. For instance, here you could explain how to set up your project in AWS and provide helpful screenshots of the process.
For run this project, you should install:
pytorch


## Dataset

### Overview
**TODO**: Explain about the data you are using and where you got it from.
The Dataset is called  Amazon Bin Image Dataset, this is a special dataset that contains almost 500000 images of bins.
For this project, the sagemaker.ipynb has the necessary statements for download a subset of it. 

### Access
**TODO**: Explain how you are accessing the data in AWS and how you uploaded it
I accessed to this data by  downloanding it from amazon s3 with this path: 
    - s3://aft-vbi-pds/bin-images
When the download process finished, I splitted it into 3 subsets:
    -- train
    -- val
    -- test
after this, I could upload to s3 by executing:
    - aws s3 cp  BinImages  s3://lucialedezmacapstoneproject/BinImages/  --recursive
  in the notebook.

## Model Training
**TODO**: What kind of model did you choose for this experiment and why? Give an overview of the types of hyperparameters that you specified and why you chose them. Also remember to evaluate the performance of your model.
I choosed the resnet50, because it is a Deep Convolutional Neural Network,  this kind of onetwork is oriented to image recognition.  since I have a classification problem, I need to recognize images into one of five classes. 


## Machine Learning Pipeline
**TODO:** Explain your project pipeline.

## Standout Suggestions
**TODO (Optional):** This is where you can provide information about any standout suggestions that you have attempted.
