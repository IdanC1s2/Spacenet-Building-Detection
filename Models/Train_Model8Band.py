# region imports
import os
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
import tensorflow as tf
import numpy as np
from UNET_Model import Build_UNET_Model
from Set_Iterators import Data_Iterator_8Band
from Utilities import save_model_history, visualize_history,\
    merge_histories_in_directory
from tensorflow.keras.callbacks import EarlyStopping
from show_results import show_masks
# endregion


# 8 Bands model:
# Set paths:
train_image_dir = 'E:/Spacenet Database/Project Database/Training Data/image_8B/img'
train_mask_dir = 'E:/Spacenet Database/Project Database/Training Data/mask/img'
val_image_dir = 'E:/Spacenet Database/Project Database/Validation Data/image_8B/img'
val_mask_dir = 'E:/Spacenet Database/Project Database/Validation Data/mask/img'
test_image_dir = 'E:/Spacenet Database/Project Database/Test Data/image_8B/img'
test_mask_dir = 'E:/Spacenet Database/Project Database/Test Data/mask/img'


# Set Data Iterators:
train_generator = Data_Iterator_8Band(batch_size=16, img_size=(128, 128),
                                      input_img_dir=train_image_dir, mask_img_dir=train_mask_dir,
                                      augment=True)

val_generator = Data_Iterator_8Band(batch_size=16, img_size=(128, 128),
                                    input_img_dir=val_image_dir, mask_img_dir=val_mask_dir)

test_generator = Data_Iterator_8Band(batch_size=16, img_size=(128, 128),
                                     input_img_dir=test_image_dir, mask_img_dir=test_mask_dir)
print('Data iterators were set.')

##########################################
# Set the model for the first time:

# Using iou_loss on the first epochs tends most of the times to get stuck
# in a local minimum where all of the weights are 0, thus predicting black masks.
# Therefore, to get past this local minimum, we use BinaryCrossEntropy Loss
# for the first 5 epochs, where we are guaranteed to have global minimum.
# Once the model is beyond this local minimum, we can use the min_iou loss, since
# maximizing the mean iou is what we're after.

model_first_5Epochs = Build_UNET_Model(128, 128, 8, Adam_lr=0.001, loss='BinaryCrossEntropy')
history1 = model_first_5Epochs.fit(x=train_generator, epochs=5, validation_data=val_generator, verbose=1)
save_model_history(history1, 'history1', dir='Models/Architecture6')
model_first_5Epochs.save_weights('Models/Architecture6/Model5Epochs_weights.h5', save_format='h5')
show_masks(model_first_5Epochs, save_path='Models/Architecture6/Masks_5_Epochs')
##########################################

# Next - we load the saved model - and start using our custom mean_iou loss:
model = Build_UNET_Model(128, 128, 8, Adam_lr=0.001)
model.load_weights('Models/Architecture6/Model5Epochs_weights.h5')


# Train Model and Save:
cb = EarlyStopping(monitor='my_mean_iou', mode='max', patience=10, restore_best_weights=True)
history2 = model.fit(x=train_generator, epochs=80, validation_data=val_generator, verbose=1)
save_model_history(history2, 'history2', dir='Models/Architecture6')
model.save_weights('Models/Architecture6/Model80Epochs_weights.h5', save_format='h5')
show_masks(model, save_path='Models/Architecture6/Masks_80_Epochs')

# Visualize History:
history = merge_histories_in_directory('Models/Architecture6')
visualize_history(history)
