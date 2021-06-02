# region Imports
import os
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, concatenate, \
     Conv2DTranspose, MaxPooling2D, Dropout, Input, Lambda
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras import backend as K
# endregion


def Build_UNET_Model(img_height=128, img_width=128, img_channels=3, Adam_lr=0.001, loss='iou'):
    if img_channels == 3:
        model = UNET_Model_3B(img_height, img_width, img_channels)
    if img_channels == 8:
        model = UNET_Model_8B(img_height, img_width, img_channels)


    if loss == 'BinaryCrossEntropy':
        model.compile(optimizer=Adam(Adam_lr), loss=BinaryCrossentropy(),
                      metrics=[my_mean_iou, my_mean_iou_hard_prediction])

    if loss == 'iou':
        model.compile(optimizer=Adam(Adam_lr), loss=my_mean_iou_loss,
                      metrics=[my_mean_iou, my_mean_iou_hard_prediction])

    return model


def UNET_Model_3B(img_height=128, img_width=128, img_channels=3):
    # model = Sequential()

    # Contraction Path
    inputs = Input(shape=(img_height,img_width,img_channels))
    inputs_normalized = Lambda(lambda x: x / 255.0)(inputs)
    c1 = Conv2D(16, (3, 3), padding='same', activation='relu', kernel_initializer='he_normal')(inputs_normalized)
    c1 = Dropout(0.1)(c1)
    c1 = Conv2D(16, (3, 3), padding='same', activation='relu', kernel_initializer='he_normal')(c1)
    p1 = MaxPooling2D(pool_size=(2, 2))(c1)

    c2 = Conv2D(32, (3, 3), padding='same', activation='relu', kernel_initializer='he_normal')(p1)
    c2 = Dropout(0.1)(c2)
    c2 = Conv2D(32, (3, 3), padding='same', activation='relu', kernel_initializer='he_normal')(c2)
    p2 = MaxPooling2D(pool_size=(2, 2))(c2)

    c3 = Conv2D(64, (3, 3), padding='same', activation='relu', kernel_initializer='he_normal')(p2)
    c3 = Dropout(0.2)(c3)
    c3 = Conv2D(64, (3, 3), padding='same', activation='relu', kernel_initializer='he_normal')(c3)
    p3 = MaxPooling2D(pool_size=(2, 2))(c3)

    c4 = Conv2D(128, (3, 3), padding='same', activation='relu', kernel_initializer='he_normal')(p3)
    c4 = Dropout(0.2)(c4)
    c4 = Conv2D(128, (3, 3), padding='same', activation='relu', kernel_initializer='he_normal')(c4)
    p4 = MaxPooling2D(pool_size=(2, 2))(c4)

    c5 = Conv2D(256, (3, 3), padding='same', activation='relu', kernel_initializer='he_normal')(p4)
    c5 = Dropout(0.3)(c5)
    c5 = Conv2D(256, (3, 3), padding='same', activation='relu', kernel_initializer='he_normal')(c5)

    # Expansive path:
    u6 = Conv2DTranspose(128, (2, 2), strides=(2,2), padding='same')(c5)
    u6 = concatenate([u6, c4])
    c6 = Conv2D(128, (3, 3), padding='same', activation='relu', kernel_initializer='he_normal')(u6)
    c6 = Dropout(0.2)(c6)
    c6 = Conv2D(128, (3, 3), padding='same', activation='relu', kernel_initializer='he_normal')(c6)

    u7 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c6)
    u7 = concatenate([u7, c3])
    c7 = Conv2D(64, (3, 3), padding='same', activation='relu', kernel_initializer='he_normal')(u7)
    c7 = Dropout(0.2)(c7)
    c7 = Conv2D(64, (3, 3), padding='same', activation='relu', kernel_initializer='he_normal')(c7)

    u8 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(c7)
    u8 = concatenate([u8, c2])
    c8 = Conv2D(32, (3, 3), padding='same', activation='relu', kernel_initializer='he_normal')(u8)
    c8 = Dropout(0.1)(c8)
    c8 = Conv2D(32, (3, 3), padding='same', activation='relu', kernel_initializer='he_normal')(c8)

    u9 = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same')(c8)
    u9 = concatenate([u9, c1], axis=3)
    c9 = Conv2D(16, (3, 3), padding='same', activation='relu', kernel_initializer='he_normal')(u9)
    c9 = Dropout(0.1)(c9)
    c9 = Conv2D(16, (3, 3), padding='same', activation='relu', kernel_initializer='he_normal')(c9)

    outputs = Conv2D(1, (1, 1), activation='sigmoid')(c9)

    model = Model(inputs=inputs, outputs=outputs)

    return model



def UNET_Model_8B(img_height=128, img_width=128, img_channels=3):
    # model = Sequential()

    # Contraction Path
    inputs = Input(shape=(img_height,img_width,img_channels))
    inputs_normalized = Lambda(lambda x: x / 255.0)(inputs)
    c1 = Conv2D(32, (3, 3), padding='same', activation='relu', kernel_initializer='he_normal')(inputs_normalized)
    c1 = Dropout(0.1)(c1)
    c1 = Conv2D(32, (3, 3), padding='same', activation='relu', kernel_initializer='he_normal')(c1)
    p1 = MaxPooling2D(pool_size=(2, 2))(c1)

    c2 = Conv2D(64, (3, 3), padding='same', activation='relu', kernel_initializer='he_normal')(p1)
    c2 = Dropout(0.1)(c2)
    c2 = Conv2D(64, (3, 3), padding='same', activation='relu', kernel_initializer='he_normal')(c2)
    p2 = MaxPooling2D(pool_size=(2, 2))(c2)

    c3 = Conv2D(128, (3, 3), padding='same', activation='relu', kernel_initializer='he_normal')(p2)
    c3 = Dropout(0.2)(c3)
    c3 = Conv2D(128, (3, 3), padding='same', activation='relu', kernel_initializer='he_normal')(c3)
    p3 = MaxPooling2D(pool_size=(2, 2))(c3)

    c4 = Conv2D(256, (3, 3), padding='same', activation='relu', kernel_initializer='he_normal')(p3)
    c4 = Dropout(0.2)(c4)
    c4 = Conv2D(256, (3, 3), padding='same', activation='relu', kernel_initializer='he_normal')(c4)
    p4 = MaxPooling2D(pool_size=(2, 2))(c4)

    c5 = Conv2D(512, (3, 3), padding='same', activation='relu', kernel_initializer='he_normal')(p4)
    c5 = Dropout(0.3)(c5)
    c5 = Conv2D(512, (3, 3), padding='same', activation='relu', kernel_initializer='he_normal')(c5)

    # Expansive path:
    u6 = Conv2DTranspose(256, (2, 2), strides=(2,2), padding='same')(c5)
    u6 = concatenate([u6, c4])
    c6 = Conv2D(256, (3, 3), padding='same', activation='relu', kernel_initializer='he_normal')(u6)
    c6 = Dropout(0.2)(c6)
    c6 = Conv2D(256, (3, 3), padding='same', activation='relu', kernel_initializer='he_normal')(c6)

    u7 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c6)
    u7 = concatenate([u7, c3])
    c7 = Conv2D(128, (3, 3), padding='same', activation='relu', kernel_initializer='he_normal')(u7)
    c7 = Dropout(0.2)(c7)
    c7 = Conv2D(128, (3, 3), padding='same', activation='relu', kernel_initializer='he_normal')(c7)

    u8 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c7)
    u8 = concatenate([u8, c2])
    c8 = Conv2D(64, (3, 3), padding='same', activation='relu', kernel_initializer='he_normal')(u8)
    c8 = Dropout(0.1)(c8)
    c8 = Conv2D(64, (3, 3), padding='same', activation='relu', kernel_initializer='he_normal')(c8)

    u9 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(c8)
    u9 = concatenate([u9, c1], axis=3)
    c9 = Conv2D(32, (3, 3), padding='same', activation='relu', kernel_initializer='he_normal')(u9)
    c9 = Dropout(0.1)(c9)
    c9 = Conv2D(32, (3, 3), padding='same', activation='relu', kernel_initializer='he_normal')(c9)

    outputs = Conv2D(1, (1, 1), activation='sigmoid')(c9)

    model = Model(inputs=inputs, outputs=outputs)

    return model


#################################


def my_mean_iou(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    y_true_f = K.cast(y_true_f, dtype='float32')
    y_pred_f = K.cast(y_pred_f, dtype='float32')
    intersection = K.sum(y_true_f * y_pred_f)
    return (intersection + K.epsilon()) / (K.sum(y_true_f) + K.sum(y_pred_f) - intersection + K.epsilon())


def my_mean_iou_loss(y_true, y_pred):
    return -my_mean_iou(y_true, y_pred)

def my_mean_iou_hard_prediction(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    y_true_f = K.cast(K.greater(y_true_f, .5), dtype='float32')
    y_pred_f = K.cast(K.greater(y_pred_f, .5), dtype='float32')  # .5 is the threshold
    intersection = K.sum(y_true_f * y_pred_f)
    return (intersection + K.epsilon()) / (K.sum(y_true_f) + K.sum(y_pred_f) - intersection + K.epsilon())







