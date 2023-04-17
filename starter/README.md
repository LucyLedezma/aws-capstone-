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
The Dataset is called  Amazon Bin Image Dataset, this is a special dataset that contains almost 500000 images of bins, the classes are presented in folders called : 1, 2, 3 , 4 and 5 ; corresponding to the number of objects in  each bin.
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
- batch-size :  I choose this because  it has a direct relation with the accuray of the model. This is the number of samples in a training step.
- lr (learning rate): I choose to tune this hyperparameter because,  it is the  step size at each training iteration while moving toward an optimum  of the loss function. in other words , it controls how fast the model learn.
First Values:
- batch-size: 64, I selected this number because is not too hight, I wanted the model takes only 64 samples per iteration, because  it requires less memory than using a larger batch-size. And the network could train faster with this size of samples in each iteration.
- lr: 0.009,  I choose this value  beacause  I do not want the model converges in  a suboptimal quickly, and it is not too low because I do not want the model learn slowly.

The   hyperparamters below were found by hyperparameter tuning process:
hyperparameters= { 'batch-size':  128,  'lr': 0.00254525353426351}


## Machine Learning Pipeline

    Upload Training Data: I will  upload the training data to an S3 bucket. but first I am going to download it from the source and process it.
    Model Training Script: I will write a script to train a model on that dataset, this script is called 'tran.py'.  I will select fixed hyperparameters and train a first model.
    Train in SageMaker: Finally I am going to use SageMaker to run that training script and train my model.
    


## Standout Suggestions
**TODO (Optional):** This is where you can provide information about any standout suggestions that you have attempted.
  - Hyperparameter tunning  I did Hyperparameter tunning, I wanted to improve my model performance, so with this result I retrained a new model. 
  - Model Profiling and Debugging: Futhermore, I performed this step, I really wanted to know how the model was trained and how the resources were used. for this step I wrote the train_hook.py script as the new  entry_point of the estimator.
  - Model Deployment:  The model is ready, but in the projects we have a team whose need to request the model and make predictions, so in order to demonstrate this skill I deployed the model into an enpoint.
  - Multi-instance training: I have setted multi-instance training with 5 instances, in this case the training job performed faster than previous configuration. This porcess took 2 hours to finish, less time than the previous training job without multi-instance setting (3 hours).
  
  ## Udacity Capstone Proposal
   the link below provide capstone proposal aproved by udacity reviewer:
   - https://review.udacity.com/#!/reviews/3998130
   
   ## Benchmark Model
   The  Benchmark model used was resnet18, I performed transfer learning with this model too.  I trained with the same dataset mentioned previously.
   Below we have the script name:
   - bench-mark.ipynb