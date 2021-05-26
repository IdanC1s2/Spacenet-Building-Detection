import numpy as np
from tensorflow.keras.preprocessing.image import load_img
import matplotlib.pyplot as plt
from Visualize_Mask import overlay_mask, show_mask
from PIL import Image
from Image_Preprocessing import predict_mask_for_image_3B, predict_mask_for_image_8B
from UNET_Model import *
import tensorflow as tf

def iou_np(y_true, y_pred):
    y_true = y_true/255
    y_pred = y_pred/255
    intersection = np.sum(y_true * y_pred)
    union = np.sum(y_true) + np.sum(y_pred)
    iou = (intersection + 1.0) / (union - intersection + 1.0)
    return iou


def show_masks(model, save_path= None, bands=8):
    img_path_test_8B = 'E:/Spacenet Database/Project Database/Test Data/image_8B/img/8band_AOI_1_RIO_img1122.tif'
    img_path_test_3B = 'E:/Spacenet Database/Project Database/Test Data/image_3B/img/3band_AOI_1_RIO_img1122.tif'
    mask_path_test = 'E:/Spacenet Database/Project Database/Test Data/mask/img/Poly_Mask_AOI_1_RIO_img1122.tif'
    if bands == 8:
        # Predicting for a given training image:
        test_img = np.array(Image.open(img_path_test_3B))
        test_mask = np.array(load_img(mask_path_test, color_mode='grayscale'))

        hard_mask, soft_mask = predict_mask_for_image_8B(model, image_path_8B=img_path_test_8B,
                                                         img_path_3B=img_path_test_3B)
        masked_image = overlay_mask(test_img, hard_mask)
    elif bands == 3:
        test_img = np.array(Image.open(img_path_test_3B))
        test_mask = np.array(load_img(mask_path_test, color_mode='grayscale'))
        hard_mask, soft_mask = predict_mask_for_image_3B(model, image_path=img_path_test_3B)
        masked_image = overlay_mask(test_img, hard_mask)

    # plt.figure(1)
    fig, ax = plt.subplots(2,2, figsize=(10,10))

    plt.sca(ax[0,0])
    plt.axis('off')
    plt.imshow(test_img)
    plt.title('Original Image')

    plt.sca(ax[0,1])
    plt.axis('off')
    plt.imshow(test_mask)
    plt.title('True Mask (Ground Truth)')

    plt.sca(ax[1,0])
    plt.axis('off')
    plt.imshow(masked_image)
    plt.title('Masked Image (Hard Prediction)')

    plt.sca(ax[1,1])
    plt.axis('off')
    plt.imshow(soft_mask)
    plt.title('Soft Mask (Soft Prediction)')

    if save_path != None:
        plt.savefig(save_path)

    iou_val_soft = iou_np(test_mask, soft_mask)
    print(f'iou_val_soft: {iou_val_soft:.4f}')
    iou_val_hard = iou_np(test_mask, hard_mask)
    print(f'iou_val_hard: {iou_val_hard:.4f}')

    return


# img_path_test_3B = 'E:/Spacenet Database/Project Database/Training Data/image_3B/img/3band_AOI_1_RIO_img385.tif'
# mask_path_test = 'E:/Spacenet Database/Project Database/Training Data/mask/img/Poly_Mask_AOI_1_RIO_img385.tif'
#
# test_img = np.array(Image.open(img_path_test_3B))
# test_mask = np.array(load_img(mask_path_test, color_mode='grayscale'))
#
# fig, ax = plt.subplots(1,2, figsize=(10,10))
# plt.sca(ax[0])
# plt.axis('off')
# plt.imshow(test_img)
# plt.title('Original Image')
#
# plt.sca(ax[1])
# plt.axis('off')
# plt.imshow(test_mask)
# plt.title('True Mask (Ground Truth)')
#



# # # load saved model and image.
# # model = Build_UNET_Model()
# # model.load_weights('Models/Attempt1/10Epochs.h5')
# # image_path = 'tmp/image/img/3band_AOI_1_RIO_img46.tif'
# # image = np.array(Image.open(image_path))
# # # predict hard and soft masks and show the hard one:
# pred_mask, pred_mask_soft = predict_mask_for_image(model=model, image_path=image_path)
# show_mask(pred_mask)
