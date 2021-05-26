# Spacenet-Building-Detection

## Step 1 - Download data:
'SpaceNet 1: Building Detection v1' Dataset from

AWS, using AWS CLI as explained in the following [link](https://spacenet.ai/spacenet-buildings-dataset-v1/).



## Step 2 - Preprocess the data:
In the end, we would like our preprocessed data to be stored as follows: 

Project Database:
- Training Data
  - Image_3B
    - img
  - Image_8B
    - img
  - mask
    - img
  - vector
- Validation Data
  - Image_3B
    - img
  - Image_8B
    - img
  - mask
    - img
  - vector
- Test Data
  - Image_3B
    - img
  - Image_8B
    - img
  - mask
    - img
  - vector

Where each of the Training, Validation and Test data groups will hold the 3 band images, 8 band images, and corresponding masks, seperate from each other.

The vector directories are holding .geogjson files that are used to create the corresponding mask for each 3 band image and its 8 band counterpart, and
we will not use the vectors directly for training the model.

The lower 'img' directories will hold the images, and are needed for the data iterators.

To do so, first of all we will need to split the data into the different directories - using the **Split_Data.py** script.

Next, we will create the corresponding masks for the data, using the **Create_Masks_For_Data.py** script.

A 3 band image and its corresponding mask look like:
![image](https://github.com/IdanC1s2/Spacenet-Building-Detection/blob/main/Images/Image%20and%20its%20true%20mask.png)

## Step 3 - Creating a model:
Since the dataset includes around 7000 3-band images of size (406,438,3) and their corresponding 8-band images of size (101,110,8),
I've created 2 U-NET segmentation models, and trained both of them.

The models are created at script **UNET_Model.py**.
In addition, we will be using a custom loss function of mean IOU to evaluate our model.

Since the **mean IOU loss** function is not necessarily convex, we find that most of the times the first few epochs of the model tend to get stuck in a local minimum,
where the model predicts every mask to be entirely black. 

To overcome this problem, we use the first few epochs to train the model on a **BinaryCrossEntropy loss** function which is convex guarantees converging towards a global minimum.
Right after getting the model past the local minimum, we switch the loss function to **mean IOU loss**, which is what we are after maximizing.

We can see that after we start training with the new loss function, predicted masks are now much more "decisive", predicting values either close to 1 or 0:
(Add image)


## Results
Show history of 


