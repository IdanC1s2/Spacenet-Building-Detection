# Spacenet-Building-Detection

## Step 1 - Download data:
'SpaceNet 1: Building Detection v1' Dataset from

AWS, using AWS CLI as explained in the following link:

https://spacenet.ai/spacenet-buildings-dataset-v1/



## Step 2 - Preprocess the data:
In the end, we would like our preprocessed data to be stored as follows: 
Project Database
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
