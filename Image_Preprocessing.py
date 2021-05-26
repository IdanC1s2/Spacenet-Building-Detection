from PIL import Image
from tensorflow.keras.preprocessing.image import load_img
import numpy as np
from osgeo import gdal
import cv2

def image_to_model_format_3B(image, model_input_shape):
    if isinstance(image, Image.Image):
        img_model = image.resize((model_input_shape[1], model_input_shape[0]))
        img_model = np.array(img_model, dtype='uint8')
    if isinstance(image, np.ndarray):
        img_model = Image.fromarray(image).resize((model_input_shape[1], model_input_shape[0]))
        img_model = np.array(img_model, dtype='uint8')

    img_as_model_input = np.expand_dims(img_model, axis=0)
    return img_as_model_input


def mask_from_model_output(mask_pred, img_original_size):
    img_height, img_width = img_original_size
    # Make sure we dont get values above 255:
    mask_pred = 255 * (mask_pred[0,:,:,0].clip(min=0, max=1))
    mask_pred = mask_pred.astype('uint8')
    # Resize it back to the original size:
    mask_pred = Image.fromarray(mask_pred).resize((img_width, img_height))
    soft_mask_pred = np.array(mask_pred)
    hard_mask_pred = np.where(soft_mask_pred>127, 255, 0)  # Hard classifying
    return hard_mask_pred, soft_mask_pred

def predict_mask_for_image_3B(model, image=None, image_path=None):
    """
    :param image: image could be given as a 3D np.ndarray
    :param image_path: image could be given as a path to load an image
    :param model: model we want to predict a mask with
    :return: mask_pred: the mask we predict
    """
    # Get model input image shape
    model_input_shape = model.layers[0].input_shape[0][1:3]
    if image is None:

        if image_path is None:  # No input was given
            raise ValueError('No input was given')

        else:  # If an image path was given
            try:
                img = load_img(image_path, color_mode='rgb')  # loading the image as a PIL.Image object
            except FileNotFoundError:
                print('Couldn\'t load image from the specified path')
            img_width, img_height = img.size

    elif isinstance(image, Image.Image):  # image is a PIL.Image object
        img_width, img_height = image.size
        img = image

    elif isinstance(image, np.ndarray):  # image is an np.ndarray
        img_height, img_width = image.shape[0:2]
        img = image
    else:
        raise ValueError('Input image was not given in a recognizable form')

    img_as_model_input = image_to_model_format_3B(img, model_input_shape)
    output = model.predict(x=img_as_model_input, batch_size=1)
    hard_mask_pred, soft_mask_pred = mask_from_model_output(output, (img_height, img_width))
    # hard_mask_pred, soft_mask_pred = mask_from_model_output(output, (128, 128))

    return hard_mask_pred, soft_mask_pred


def predict_mask_for_image_8B(model, image=None, image_path_8B=None, img_path_3B=None):
    # Get model input image shape
    model_input_shape = model.layers[0].input_shape[0][1:3]
    if image is None:

        if image_path_8B is None:  # No input was given
            raise ValueError('No input was given')

        else:  # If an image path was given
            try:
                img = gdal.Open(image_path_8B).ReadAsArray()  # loading the image using gdal
                img = np.moveaxis(img, 0, 2)
                img_3B = np.array(load_img(img_path_3B))
            except FileNotFoundError:
                print('Couldn\'t load image from the specified path')
            img_height, img_width = img_3B.shape[0:2]


    elif isinstance(image, np.ndarray):  # image is an np.ndarray
        img_height, img_width = image.shape[0:2]
        img = image
    else:
        raise ValueError('Input image was not given in a recognizable form')

    img_as_model_input = image_to_model_format_8B(img, model_input_shape)
    output = model.predict(x=img_as_model_input, batch_size=1)
    hard_mask_pred, soft_mask_pred = mask_from_model_output(output, (img_height, img_width))
    return hard_mask_pred, soft_mask_pred


def image_to_model_format_8B(image, model_input_shape):
    if image.dtype == 'uint16':
        img_model = (255 * (cv2.resize(image, model_input_shape) / (2 ** 16))).astype('uint8')
    if image.dtype == 'uint8':
        img_model = cv2.resize(image, model_input_shape)
    img_as_model_input = np.expand_dims(img_model, axis=0)
    return img_as_model_input




