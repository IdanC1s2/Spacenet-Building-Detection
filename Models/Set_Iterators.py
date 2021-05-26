from tensorflow.keras.utils import Sequence
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
from tensorflow.keras.preprocessing.image import load_img
import os
from natsort.natsort import natsorted
from Mean_IoU import calculate_mean_iou_of_mask
from tensorflow.keras.utils import normalize
from osgeo import gdal
from PIL import Image
import cv2


class Data_Iterator_3Band(Sequence):
    """Helper to iterate over the data (as Numpy arrays)."""

    def __init__(self, batch_size, img_size, input_img_dir, mask_img_dir, augment=False, seed=1):
        self.batch_size = batch_size
        self.img_size = img_size
        self.input_img_paths = Image_Dir_Into_List_Of_Paths(input_img_dir)
        self.target_img_paths = Image_Dir_Into_List_Of_Paths(mask_img_dir)
        # self.aug_seeds is used for getting the same augmentations per matching images and masks,
        # however, we need to change it every iteration so that our augmentations will be different
        # with every epoch. Thus we define self.aug_seeds to be a vector of random numbers generated
        # by self.seed and it is altered in the end of every __getitem__ call.
        self.seed = seed
        np.random.seed(seed)
        self.augment = augment
        if self.augment:
            self.aug_seeds = np.random.randint(1, 10000, self.batch_size)
            # self.generator = ImageDataGenerator(width_shift_range=0.1, height_shift_range=0.1,
            #                                     rotation_range=5, dtype='uint8', zoom_range=0.1,
            #                                     fill_mode='constant', cval=0,
            #                                     horizontal_flip=True, vertical_flip=True)

            # self.generator = ImageDataGenerator(dtype='uint8', rotation_range=5,
            #                                     fill_mode='constant', cval=0,
            #                                     horizontal_flip=True, vertical_flip=True)
            self.generator = ImageDataGenerator(dtype='uint8', vertical_flip=True, horizontal_flip=True)
    def __len__(self):
        return len(self.target_img_paths) // self.batch_size

    def __getitem__(self, idx):
        """Returns tuple (input, target) correspond to batch #idx."""
        i = idx * self.batch_size
        batch_input_img_paths = self.input_img_paths[i: i + self.batch_size]
        batch_target_img_paths = self.target_img_paths[i: i + self.batch_size]

        # Create input batch with self.batch_size images:
        x = np.zeros((self.batch_size,) + self.img_size + (3,), dtype='uint8')
        for j, path in enumerate(batch_input_img_paths):
            img = load_img(path, target_size=self.img_size)
            if self.augment:
                img = np.expand_dims(img, axis=0)
                # augment the single image:
                aug_img = self.generator.flow(img, seed=self.aug_seeds[j], batch_size=1)[0]
                x[j] = aug_img
            else:
                x[j] = img


        # Create mask (target) batch with self.batch_size images:
        y = np.zeros((self.batch_size,) + self.img_size + (1,), dtype='uint8')
        for j, path in enumerate(batch_target_img_paths):
            mask = load_img(path, target_size=self.img_size, color_mode='grayscale')
            if self.augment:
                mask = np.expand_dims(mask, axis=[0,3])
                aug_mask = self.generator.flow(mask, seed=self.aug_seeds[j], batch_size=1)[0]
                y[j] = aug_mask
            else:
                y[j] = np.expand_dims(mask, 2)

            # make it into categorical mask of 0's and 1's:
            y[j] = y[j] / 255
        # Change aug_seed for next iterations to have different augmentations
        self.aug_seeds = np.random.randint(1, 10000, self.batch_size)
        return x, y




class Data_Iterator_8Band(Sequence):
    """Helper to iterate over the data (as Numpy arrays)."""

    def __init__(self, batch_size, img_size, input_img_dir, mask_img_dir, augment=False, seed=1):
        self.batch_size = batch_size
        self.img_size = img_size
        self.input_img_paths = Image_Dir_Into_List_Of_Paths(input_img_dir)
        self.target_img_paths = Image_Dir_Into_List_Of_Paths(mask_img_dir)
        # self.aug_seeds is used for getting the same augmentations per matching images and masks,
        # however, we need to change it every iteration so that our augmentations will be different
        # with every epoch. Thus we define self.aug_seeds to be a vector of random numbers generated
        # by self.seed and it is altered in the end of every __getitem__ call.
        self.seed = seed
        np.random.seed(seed)
        self.augment = augment
        if self.augment:
            self.aug_seeds = np.random.randint(1, 10000, self.batch_size)
            # self.generator = ImageDataGenerator(width_shift_range=0.1, height_shift_range=0.1,
            #                                     rotation_range=5, dtype='uint8', zoom_range=0.1,
            #                                     fill_mode='constant', cval=0,
            #                                     horizontal_flip=True, vertical_flip=True)

            # self.generator = ImageDataGenerator(dtype='uint8', rotation_range=5,
            #                                     fill_mode='constant', cval=0,
            #                                     horizontal_flip=True, vertical_flip=True)
            self.generator = ImageDataGenerator(dtype='uint8', vertical_flip=True, horizontal_flip=True)
    def __len__(self):
        return len(self.target_img_paths) // self.batch_size

    def __getitem__(self, idx):
        """Returns tuple (input, target) correspond to batch #idx."""
        i = idx * self.batch_size
        batch_input_img_paths = self.input_img_paths[i: i + self.batch_size]
        batch_target_img_paths = self.target_img_paths[i: i + self.batch_size]

        # Create input batch with self.batch_size images:
        x = np.zeros((self.batch_size,) + self.img_size + (8,), dtype='uint8')
        for j, path in enumerate(batch_input_img_paths):
            img = np.moveaxis(gdal.Open(path).ReadAsArray(), 0, 2)
            img = (255 * (cv2.resize(img, self.img_size) / (2**16))).astype('uint8')
            if self.augment:
                img = np.expand_dims(img, axis=0)
                # augment the single image:
                aug_img = self.generator.flow(img, seed=self.aug_seeds[j], batch_size=1)[0]
                x[j] = aug_img
            else:
                x[j] = img


        # Create mask (target) batch with self.batch_size images:
        y = np.zeros((self.batch_size,) + self.img_size + (1,), dtype='uint8')
        for j, path in enumerate(batch_target_img_paths):
            mask = load_img(path, target_size=self.img_size, color_mode='grayscale')
            if self.augment:
                mask = np.expand_dims(mask, axis=[0,3])
                aug_mask = self.generator.flow(mask, seed=self.aug_seeds[j], batch_size=1)[0]
                y[j] = aug_mask
            else:
                y[j] = np.expand_dims(mask, 2)

            # make it into categorical mask of 0's and 1's:
            y[j] = y[j] / 255
        # Change aug_seed for next iterations to have different augmentations
        self.aug_seeds = np.random.randint(1, 10000, self.batch_size)
        return x, y



# class Data_Iterator_Augmentation(Sequence):
#     """Helper to iterate over the data (as Numpy arrays)."""
#
#     def __init__(self, batch_size, img_size, input_img_dir, mask_img_dir, seed):
#         self.batch_size = batch_size
#         self.img_size = img_size
#         self.input_img_paths = Image_Dir_Into_List_Of_Paths(input_img_dir)
#         self.target_img_paths = Image_Dir_Into_List_Of_Paths(mask_img_dir)
#         self.augment = augment
#
#         self.generator = ImageDataGenerator(width_shift_range=0.1, height_shift_range=0.1,
#                                                 rotation_range=5, dtype='uint8')
#         self.mask_augmentor = self.generator.flow()
#     def __len__(self):
#         return len(self.target_img_paths) // self.batch_size
#
#     def __getitem__(self, idx):
#         """Returns tuple (input, target) correspond to batch #idx."""
#         i = idx * self.batch_size
#         batch_input_img_paths = self.input_img_paths[i: i + self.batch_size]
#         batch_target_img_paths = self.target_img_paths[i: i + self.batch_size]
#
#         # Create input batch with self.batch_size images:
#         x = np.zeros((self.batch_size,) + self.img_size + (3,), dtype='uint8')
#         for j, path in enumerate(batch_input_img_paths):
#             img = load_img(path, target_size=self.img_size)
#             x[j] = img
#
#         # Create mask (target) batch with self.batch_size images:
#         y = np.zeros((self.batch_size,) + self.img_size + (1,), dtype='uint8')
#         for j, path in enumerate(batch_target_img_paths):
#             img = load_img(path, target_size=self.img_size, color_mode='grayscale')
#             y[j] = np.expand_dims(img, 2)
#             # # Ground truth labels are 1, 2, 3. Subtract one to make them 0, 1, 2:
#             # y[j] -= 1
#             # make it into categorical mask of 0's and 1's:
#             y[j] = y[j] / 255
#         return x, y

def Image_Dir_Into_List_Of_Paths(dir):
    src_list = natsorted(os.listdir(dir))
    path_list = [os.path.join(dir, file_src) for file_src in src_list]
    return path_list
