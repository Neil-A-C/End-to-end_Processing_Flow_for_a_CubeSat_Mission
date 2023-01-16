import os
import cv2 as cv
import sys
from time import process_time
import csv
import numpy as np
from fmask.cmdline import sentinel2Stacked
import DisplayingData

def cloudClassification(safe_dir,out_img_file):
    """
        This function runs a command line script that comes with the FMask
        library. Documentation for the library in question can be found at:
            https://www.pythonfmask.org/en/latest/
        It takes raw data of a Sentinel-2 Level 1-C image and creates a
        classification according to the FMask algorithm.

        **IMPORTANT** the default output image fromat is an 'HFA' image, which is
        pretty ureadable to change the output file type from the FMask script, and
        make this script work, you need to change the RIOS driver default in your
        environment using the command "export RIOS_DFLT_DRIVER='GTiff'".

        Arguments:
            *safe_dir - The name of the .SAFE directory where the Sentinel-2
                        data is stored.
            *out_img_file - The file path for the output of the script to be
                            written to.
    """
    sentinel2Stacked.mainRoutine(['--safedir', safe_dir, '-o', out_img_file])
    print('FMask Cloud Classification Complete!')

def createMask(in_file):
    """
        This function creates a binary cloud mask from an FMask classification.

        Arguments:
            *in_file - The file path to the FMask classification image.
        Returns:
            *A boolean numpy array where cloud is marked as 'False'
             and non-cloud is marked as 'True'.
    """
    in_img = cv.imread(cv.samples.findFile(in_file)) # read the FMask classification file

    out_img = np.zeros((in_img.shape[0],in_img.shape[1]),dtype=np.uint8)  # initialise the output array

    # if either the green or red channels of the classification image are non-zero, set that pixel to white for the output
    out_img = np.maximum(in_img[:,:,1],in_img[:,:,2]) # put the greatest values of the green and red channels of the into out_img
    out_img[out_img != 0] = 255

    return ~out_img.astype(np.bool_)

if __name__=="__main__":

    # make a directory to put the outputs into
    out_path = 'CloudMasking_Test/'
    if not os.path.exists(out_path):
        os.mkdir(out_path)

    safe_dir = 'Test.SAFE' # name of the test input .SAFE directory
    cloud_classification_file = out_path+'classification.tif' # specify the name of the classification file

    cloudClassification(safe_dir,cloud_classification_file) # run FMask classification

    # extract the image from the .SAFE directory
    image = DisplayingData.extractBands(safe_dir,(5490,5490)) # output an RGB image of the data
    DisplayingData.createImage(out_path+'RGB.png',image)
    np.save('full_image.npy',image)
    DisplayingData.propertiesTest(image,'image')

    # create a mask from the FMask classification
    cloud_mask = createMask(cloud_classification_file)
    DisplayingData.createBinaryImage(out_path+'cloud_mask.png',cloud_mask)
    np.save('cloud_mask.npy',cloud_mask)
    DisplayingData.propertiesTest(cloud_mask,'cloud_mask')

    # output the masked image
    DisplayingData.createImage(out_path+'masked_RGB.png',image,cloud_mask)
