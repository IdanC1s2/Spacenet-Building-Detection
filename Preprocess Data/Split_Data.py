import os
from pathlib import Path
from random import shuffle
from shutil import copyfile


def Split_Training_Data_To_Pathfiles(training_folder, dst_folder, train_ratio, val_ratio, test_ratio):
    ratios_sum = (train_ratio + val_ratio + test_ratio)
    if ratios_sum != 1:
        train_ratio = train_ratio / ratios_sum
        val_ratio = val_ratio / ratios_sum
        test_ratio = test_ratio / ratios_sum

    image_names = os.listdir(training_folder)
    shuffle(image_names)

    # Remove path files if they already exist:
    for file in ['test.txt', 'training.txt', 'validation.txt']:
        if file in os.listdir(dst_folder):
            os.remove(dst_folder + '/' + file)

    # Setting the training data file
    cutting_idx1 = int(train_ratio * len(image_names))
    with open(dst_folder + '/training.txt', 'w') as f:
        for i, image_name in enumerate(image_names[0:cutting_idx1]):
            if i != 0:
                f.write('\n')
            f.write(image_name)

    # Setting the validation data file:
    cutting_idx2 = int(cutting_idx1 + 1 + val_ratio * len(image_names))
    with open(dst_folder + '/validation.txt', 'w') as f:
        for i, image_name in enumerate(image_names[cutting_idx1:cutting_idx2]):
            if i != 0:
                f.write('\n')
            f.write(image_name)

    # Setting the test data file:
    with open(dst_folder + '/test.txt', 'w') as f:
        for i, image_name in enumerate(image_names[cutting_idx2:]):
            if i != 0:
                f.write('\n')
            f.write(image_name)


def Split_Training_Data_To_Directories(training_image_folder_3B, training_image_folder_8B,
                                       dst_img_folders_3B, dst_img_folders_8B,
                                       training_vector_folder,
                                       dst_vector_folders, train_ratio, val_ratio, test_ratio):
    """

    :param training_image_folder_3B: folder that contains the 3band tiff images
    :param dst_img_folders_3B: a tuple of path objects (train_image_path, val_image_path, test_image_path)
    where these values contain the paths to the directories of the train, validation, and test images
    where we want to store our split 3band image data.
    :param dst_img_folders_8B: a tuple of path objects (train_image_path, val_image_path, test_image_path)
    where these values contain the paths to the directories of the train, validation, and test images
    where we want to store our split 8band image data.
    :param training_vector_folders: folder that contains the vector .geojson files
    :param dst_vector_folders: tuple of path objects ( train_vector_path, val_vector_path, test_vector_path)
     where these values contain the paths to the directories of the train, validation, and test images
     where we want to store our split vector data.
    :param train_ratio: the part of the data which is to be considered as training samples
    :param val_ratio: the part of the data which is to be considered as validation samples
    :param test_ratio: the part of the data which is to be considered as test samples

    """
    ratios_sum = (train_ratio + val_ratio + test_ratio)
    if ratios_sum != 1:
        train_ratio = train_ratio / ratios_sum
        val_ratio = val_ratio / ratios_sum
        test_ratio = test_ratio / ratios_sum

    dst_train_image_path_3B, dst_val_image_path_3B, dst_test_image_path_3B = dst_img_folders_3B
    dst_train_image_path_8B, dst_val_image_path_8B, dst_test_image_path_8B = dst_img_folders_8B
    dst_train_vector_path, dst_val_vector_path, dst_test_vector_path = dst_vector_folders

    image_names_3B = os.listdir(training_image_folder_3B)
    image_names_8B = os.listdir(training_image_folder_8B)
    shuffle(image_names_3B)

    cutting_idx1 = int(train_ratio * len(image_names_3B))
    cutting_idx2 = int(cutting_idx1 + 1 + val_ratio * len(image_names_3B))

    # Copying images and their corresponding geojson files to train folder:
    for image_src in image_names_3B[0:cutting_idx1]:
        # Copying images first:
        copyfile(training_image_folder_3B.joinpath(image_src), dst_train_image_path_3B.joinpath(image_src))
        copyfile(training_image_folder_8B.joinpath('8' + image_src[1:]), dst_train_image_path_8B.joinpath('8' + image_src[1:]))

        # Copying geojson files:
        vector_src = image_src.replace('3band_', 'Geo_').replace('.tif', '.geojson')
        vector_path = training_vector_folder.joinpath(vector_src)
        copyfile(vector_path, dst_train_vector_path.joinpath(vector_src))

    # Copying images and their corresponding geojson files to validation folder:
    for image_src in image_names_3B[cutting_idx1:cutting_idx2]:
        # Copying images first:
        copyfile(training_image_folder_3B.joinpath(image_src), dst_val_image_path_3B.joinpath(image_src))
        copyfile(training_image_folder_8B.joinpath('8' + image_src[1:]), dst_val_image_path_8B.joinpath('8' + image_src[1:]))
        # Copying geojson files:
        vector_src = image_src.replace('3band_', 'Geo_').replace('.tif', '.geojson')
        vector_path = training_vector_folder.joinpath(vector_src)
        copyfile(vector_path, dst_val_vector_path.joinpath(vector_src))

    # Copying images and their corresponding geojson files to test folder:
    for image_src in image_names_3B[cutting_idx2:]:
        # Copying images first:
        copyfile(training_image_folder_3B.joinpath(image_src), dst_test_image_path_3B.joinpath(image_src))
        copyfile(training_image_folder_8B.joinpath('8' + image_src[1:]), dst_test_image_path_8B.joinpath('8' + image_src[1:]))
        # Copying geojson files:
        vector_src = image_src.replace('3band_', 'Geo_').replace('.tif', '.geojson')
        vector_path = training_vector_folder.joinpath(vector_src)
        copyfile(vector_path, dst_test_vector_path.joinpath(vector_src))
    return


if __name__ == '__main__':
    Project_Datasets_Path = Path('E:/Spacenet Database/Project Database')
    train_image_Path_3B = Path('E:/Spacenet Database/Train/3band')
    train_image_Path_8B = Path('E:/Spacenet Database/Train/8band')
    train_vector_Path = Path('E:/Spacenet Database/Train/geojson')

    # Creating folders of
    image_folders_3B = []
    image_folders_8B = []
    vector_folders = []
    for folder1 in ['Training Data', 'Validation Data', 'Test Data']:
        vectors_path = Project_Datasets_Path.joinpath(folder1).joinpath('vector')
        os.makedirs(vectors_path, exist_ok=True)
        vector_folders.append(vectors_path)
        for folder2 in ['image_3B', 'image_8B', 'mask']:
            images_path = Project_Datasets_Path.joinpath(folder1).joinpath(folder2).joinpath('img')
            os.makedirs(images_path, exist_ok=True)
            if folder2 == 'image_3B':
                image_folders_3B.append(images_path)
            if folder2 == 'image_8B':
                image_folders_8B.append(images_path)

    Split_Training_Data_To_Directories(training_image_folder_3B=train_image_Path_3B,
                                       training_image_folder_8B =train_image_Path_8B,
                                       dst_img_folders_3B=image_folders_3B,
                                       dst_img_folders_8B=image_folders_8B,
                                       training_vector_folder=train_vector_Path, dst_vector_folders=vector_folders,
                                       train_ratio=0.7, val_ratio=0.2, test_ratio=0.1)
