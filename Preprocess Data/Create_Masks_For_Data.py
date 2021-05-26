from osgeo import gdal, ogr
import os
from pathlib import Path
from natsort.natsort import natsorted


def create_poly_mask(rasterSrc, vectorSrc, npDistFileName='',
                     burn_values=255):
    """

    :param rasterSrc: path to raster to be emulated
    :param vectorSrc: path to vector corresponding to the raster
    :param npDistFileName: optional - path to which we save the poly mask as tif image
    :param burn_values: 255 - pixel value we burn in the polygons area
    :return: a mask_image which is a 2D array.
    """


    # open source vector file that truth data
    source_ds = ogr.Open(vectorSrc)
    source_layer = source_ds.GetLayer()

    # extract data from src Raster File to be emulated
    # open raster file that is to be emulated
    srcRas_ds = gdal.Open(rasterSrc)
    # get image width and height:
    cols = srcRas_ds.RasterXSize
    rows = srcRas_ds.RasterYSize

    if npDistFileName == '':
        dstPath = "tmp.tif"
    else:
        dstPath = npDistFileName

    # Create first raster memory layer, units are pixels
    # Change output to geotiff instead of memory
    memdrv = gdal.GetDriverByName('GTiff')
    dst_ds = memdrv.Create(dstPath, cols, rows, 1, gdal.GDT_Byte,
                           options=['COMPRESS=LZW'])
    dst_ds.SetGeoTransform(srcRas_ds.GetGeoTransform())
    dst_ds.SetProjection(srcRas_ds.GetProjection())
    gdal.RasterizeLayer(dst_ds, [1], source_layer, burn_values=[burn_values], options=["BURN_VALUE_FROM=Z"])
    dst_ds = 0

    mask_image = gdal.Open(dstPath)
    mask_image = mask_image.ReadAsArray()

    if npDistFileName == '':
        os.remove(dstPath)

    return mask_image

print('checkpoint')

def Create_Masks_For_Dataset(raster_Src_dir, vector_Src_dir, dst_dir):

    # Create dst_dir if non existent:
    os.makedirs(dst_dir, exist_ok=True)

    # Iterating over the rasters with their corresponding vectors, and creating
    # a poly_mask for each raster in dst_dir
    for (raster, vector) in zip(natsorted(os.listdir(raster_Src_dir)), natsorted(os.listdir(vector_Src_dir))):
        rasterSrc = raster_Src_dir + '/' + raster
        vectorSrc = vector_Src_dir + '/' + vector
        poly_mask_dst = dst_dir + '/' + raster.replace('3band', 'Poly_Mask')
        create_poly_mask(rasterSrc=rasterSrc, vectorSrc=vectorSrc,
                         npDistFileName=poly_mask_dst, burn_values=255)


if __name__ == '__main__':

    # We want to create Poly Mask for each of our 3band images in the training, val, and test sets.
    # Lets define our paths first:
    # Training set:
    rasterSrc_dir_train = 'E:/Spacenet Database/Project Database/Training Data/image_3B/img'
    vectorSrc_dir_train = 'E:/Spacenet Database/Project Database/Training Data/vector'
    dst_dir_train = 'E:/Spacenet Database/Project Database/Training Data/mask/img'
    Create_Masks_For_Dataset(raster_Src_dir=rasterSrc_dir_train, vector_Src_dir=vectorSrc_dir_train,
                             dst_dir=dst_dir_train)
    # Validation set:
    rasterSrc_dir_val = 'E:/Spacenet Database/Project Database/Validation Data/image_3B/img'
    vectorSrc_dir_val = 'E:/Spacenet Database/Project Database/Validation Data/vector'
    dst_dir_val = 'E:/Spacenet Database/Project Database/Validation Data/mask/img'
    Create_Masks_For_Dataset(raster_Src_dir=rasterSrc_dir_val, vector_Src_dir=vectorSrc_dir_val,
                             dst_dir=dst_dir_val)
    # Test set
    rasterSrc_dir_test = 'E:/Spacenet Database/Project Database/Test Data/image_3B/img'
    vectorSrc_dir_test = 'E:/Spacenet Database/Project Database/Test Data/vector'
    dst_dir_test = 'E:/Spacenet Database/Project Database/Test Data/mask/img'
    Create_Masks_For_Dataset(raster_Src_dir=rasterSrc_dir_test, vector_Src_dir=vectorSrc_dir_test,
                             dst_dir=dst_dir_test)
