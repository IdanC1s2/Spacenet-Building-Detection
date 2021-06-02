# Spacenet-Building-Detection
## Dependencies:
Osgeo.Gdal\
PIL\
OpenCV\
Tensorflow.Keras

## Step 1 - Download data:
'SpaceNet 1: Building Detection v1' Dataset from\
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

Where each of the Training, Validation and Test data groups will hold the 3 band images,\
8 band images, and corresponding masks, seperate from each other.

The vector directories are holding .geojson files that are used to create the corresponding mask for each\
3 band image and its 8 band counterpart, and we will not use the vectors directly for training the model.\
The lower 'img' directories will hold the images, and are needed for the data iterators.

To do so, first of all we will need to split the data into the different directories - using the **Split_Data.py** script.\
Next, we will create the corresponding masks for the data.\
The labeled masks are given in geotiff format - meaning that each mask is basically a polygon whose vertices\
are given by geographic coordinates. To transform these polygons into masked images, we will be using the **Create_Masks_For_Data.py** script.


A 3 band image and its corresponding mask look like:
![image](https://github.com/IdanC1s2/Spacenet-Building-Detection/blob/main/Images/Image%20and%20its%20true%20mask.png)

## Step 3 - Creating a model:
Since the dataset includes around 7000 3-band images of size (406,438,3) and their corresponding 8-band images of size (101,110,8),
I've created 2 U-NET segmentation models, and trained both of them.

The models are created at script **UNET_Model.py**.

Our UNET architecture for the 8 bands data will be as follows:
![image](https://github.com/IdanC1s2/Spacenet-Building-Detection/blob/main/Images/UNET%20Architecture%208B.jpg)

We might as well ad some dropout layers in-between some of the layers.

In addition, we will be using a custom loss function of mean IOU to evaluate our model.

Since the **mean IOU loss** function is not necessarily convex, we find that most of the times the first few epochs of the model tend to get stuck in a local minimum,
where the model predicts every mask to be entirely black. 

To overcome this problem, we use the first few epochs to train the model on a **BinaryCrossEntropy loss** function which is convex guarantees converging towards a global minimum. We end up getting a weak model that doesn't predict black masks, but can capture some of the image's characteristics:
![image](https://github.com/IdanC1s2/Spacenet-Building-Detection/blob/main/Images/Masks_5_Epoch.png)

Right after getting the model past the local minimum, we switch the loss function to **mean IOU loss**, which is what we are after maximizing.
We can see that after we start training with the new loss function, predicted masks are now much more "decisive", predicting values either close to 1 or 0, and giving much 'edgier' predictions. (As will be shown)

To evaluate our model we have 2 metrics - 'mean_iou' which measures the similarity between the true mask with the predicted mask,\
and 'mean_iou_hard' which measures the similarity between the true mask with the hard predicted mask.\
(values above 0.5 are set to 1, and below 0.5 are set to 0)\
Actually, the negative of the 'mean_iou_hard' should have benn our loss function, and minimizing it would maximize the mean IOU value\
of a true mask and its hard predicted mask, but 'mean_iou_hard' is not differentiable, so we had to use the soft version as our loss function.

For the example shown above, the model of 5 epochs of BinaryCrossEntropy loss showed nice results:\
iou_val_soft: 0.3833\
iou_val_hard: 0.4259



## Results
After 80 epochs of training the 8 band model, the model training set converged to iou value of around 0.55,\
and the validation training set converged to around 0.53.

![history](https://github.com/IdanC1s2/Spacenet-Building-Detection/blob/main/Images/history80epochs.png)

If we look at the example showed above, our model gives much better predictions now:

![image](https://github.com/IdanC1s2/Spacenet-Building-Detection/blob/main/Images/Masks_80_Epochs.png)
Together with IOU values of:\
iou_val_soft: 0.6717\
iou_val_hard: 0.6736

