import cv2 as cv
import numpy as np
import glob
import os

def propertiesTest(img,input_name,list_values=False):
    """
        This function takes a numpy array and prints useful information about
        the array to the terminal.

        Arguments:
            *img - the numpy array
            *input_name - the name given to the array in the terminal (string)
            *list_values - when true, the counts of each value will be shown
    """
    shape = img.shape
    size = img.size
    dtype = img.dtype
    values = np.unique(img)

    print("Image Name:",input_name)
    print("The datatype output from cv.imread is: ",type(img))
    print("The dimensions of the image are:",shape)
    print("The size of the image is: ",size,"pixels")
    print("The data types of the values are: ",dtype)
    print("The values in the array are: ",values)

    if list_values == True:
        for value in values:
            print("There are ",(img==value).sum()," ",value,"'s in the array")
    print()

def findGranuleDir(safedir):
    """
    Copied from 'sentinel2stacked.py' by Neil Flood:
    https://www.pythonfmask.org/en/latest/

    Search the given .SAFE directory, and find the main XML file at the GRANULE level.

    Note that this currently only works for the new-format zip files, with one
    tile per zipfile. The old ones are being removed from service, so we won't
    cope with them.

    """
    granuleDirPattern = "{}/GRANULE/L1C_*".format(safedir)
    granuleDirList = glob.glob(granuleDirPattern)
    if len(granuleDirList) == 0:
        raise fmaskerrors.FmaskFileError("Unable to find GRANULE sub-directory {}".format(granuleDirPattern))
    elif len(granuleDirList) > 1:
        dirstring = ','.join(granuleDirList)
        msg = "Found multiple GRANULE sub-directories: {}".format(dirstring)
        raise fmaskerrors.FmaskFileError(msg)

    granuleDir = granuleDirList[0]
    return granuleDir

def extractBands(safedir,target_dimensions):
    """
        Based on elements of makeStackAndAngles() from 'sentinel2stacked.py' by Neil Flood:
        https://www.pythonfmask.org/en/latest/

        This function takes the path to a .SAFE directory of Sentinel-2 Level-1C
        data and returns a numpy array with 13 channels, one for each band of
        data. The first two dimensions of the output array are determined by the
        'target_dimensions' argument, where all bands are resized to that shape.
    """
    bandList = ['B01', 'B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B8A',
        'B09', 'B10', 'B11', 'B12'] # The list of band names
    imgDir = "{}/IMG_DATA".format(findGranuleDir(safedir)) # Find the path to the image data

    full_image = np.empty((target_dimensions+(len(bandList),)),dtype = np.uint8) # initialise the output array
    for i in range(len(bandList)):
        inBandImgList = glob.glob("{}/*_{}.jp2".format(imgDir, bandList[i])) # the path to the band image file
        # make sure only one file was found by glob.glob()
        if len(inBandImgList) != 1:
            raise fmaskerrors.FmaskFileError("Cannot find input band {}".format(band))
        inBandImg = inBandImgList[0]

        band = cv.imread(inBandImg) # read the band file

        # resize the band to deired shape
        full_image[:,:,i] = cv.resize(band[:,:,0],target_dimensions,interpolation=cv.INTER_NEAREST)

    return full_image

def createImage(file_path,array,mask=None):
    """
        This function outputs an RGB image from a given numpy array of
        Sentinel-2 data. If a mask has been provided it will also be applied to
        the image, with all masked pixels (where the mask value is False) set to 0.

        Arguments:
            *file_path - the path to the output file (any file type compatible with openCV.imwrite)
            *array - The numpy array of Sentinel-2 data, where all bands have the same dimensions
            *mask - The numpy boolean array of a binary mask (optional)
    """
    in_shape = array.shape # get the shape of the input array

    out_shape = (in_shape[0],in_shape[1],3) # the shape of the output will be have the same frist two dimensions as the input, but only 3 channels
    bgr = np.empty(out_shape,dtype = np.uint8) # initialise the output array
    clahe = cv.createCLAHE() # set up Contrast Limited Adaptive Histogram Equalization

    # apply CLAHE to each band and put them in the output array
    bgr[:,:,0] = clahe.apply(array[:,:,1]) # blue
    bgr[:,:,1] = clahe.apply(array[:,:,2]) # green
    bgr[:,:,2] = clahe.apply(array[:,:,3]) # red

    # apply the mask, if provided
    if type(mask) is np.ndarray:
        mask = np.dstack([mask]*3) # match the mask to all values in the array
        bgr[~mask] = 0 # set points where mask is False to 0.

    cv.imwrite(file_path,bgr) #output the image to file_path

def createBinaryImage(file_path,array):
    """
        This function creates a binary image from a numpy array with
        'dtype=np.bool_'.

        Arguments:
            *file_path - the path to the output file (any file type compatible with openCV.imwrite)
            *array - The boolean numpy array
    """
    image = np.zeros(array.shape,dtype=np.uint8)
    image[array] = 255
    cv.imwrite(file_path,image)

if __name__=="__main__":

    # make a directory to put the outputs into
    out_dir = 'DisplayingData_Test/'
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    safe_path = 'Test.SAFE' # name of the test input .SAFE directory

    full_image = extractBands(safe_path,(5490,5490)) # create a numpy array from the .SAFE file
    propertiesTest(full_image,'full_image') #demonstrate the use of propertiesTest

    createImage(out_dir+'out_unmasked.png',full_image) # output an RGB image of the data

    # create a rectangular mask
    mask = np.zeros((full_image.shape[0],full_image.shape[1]),dtype=np.bool_)
    mask[100:-100,100:-100] = True
    createBinaryImage(out_dir+'mask.png',mask)
    propertiesTest(mask,'mask',list_values=True) # another example of the use of propertiesTest

    createImage(out_dir+'out_masked.png',full_image,mask) # output an masked version of the RGB image of the data
