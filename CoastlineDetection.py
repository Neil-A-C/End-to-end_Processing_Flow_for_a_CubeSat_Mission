import sys
import os
import cv2 as cv
import numpy as np
import skimage.filters
import DisplayingData
from matplotlib import pyplot as plt

def detectCoastline(image,mask,return_NDWI=False,return_histogram=False):
    """
        This function segments a Sentinel-2 satellite image into water and
        non-water, using Noramlised Difference Water Index (NDWI) and ISODATA
        thresholding.

        Arguments:
            *image - a numpy array of Sentinel-2 data, where all bands have the
                     same dimensions
            *mask - a boolean numpy array of the mask, whcih is being applied to
                    the image
            *return_NDWI - specifies whether a scaled version of the NDWI array
                           should be returned (boolean - default: 'False')
            *return_histogram - determines whether a histogram of the input
                                image should be returned (boolean - default:
                                'False')
        Returns:
            *label - a 2-dimensional numpy array labelling water as 255 and
                     non-water as 0
            *thresh - the threshold applied to the NDWI array to obtain the
                      label
            *scaled_NDWI - a version of the NDWI array, which has been scaled to
                           range between 0 and 255, rather than -1 to 1, is
                           returned if return_NDWI=True
    """
    # combine the image and mask into a numpy masked array for easy NDWI calculation
    masked_image = np.ma.array(image,mask=np.dstack([~mask]*image.shape[2]),dtype=np.int16)

    green = masked_image[:,:,2] # Band 3 of Sentinel-2 images is the green band.
    nir = masked_image[:,:,7] # Band 8 of Sentinel-2 images is the NIR band.
    NDWI = (green-nir)/(green+nir) # The Normalised Difference Water Index calculation.

    # scale NDWI from -1 to 1 up to 0 to 255, for output as an openCV compatible array
    if return_NDWI:
        scaled_NDWI = (NDWI + 1) * 127
        scaled_NDWI = scaled_NDWI.astype(np.uint8)

    # mask = np.ma.getmask(NDWI) # get the mask from the masked array

    # ISODATA/inter-means threshold
    # only uses the 'data' from the masked array, but applies a comprehension to skip over masked values
    thresh = skimage.filters.threshold_isodata(np.ma.getdata(NDWI)[mask])

    thresh,label = cv.threshold(NDWI,thresh,255,cv.THRESH_BINARY) # apply the threshold

    # start making a tuple of outputs
    out = (label.astype(np.uint8),thresh)

    # add NDWI to the output tuple, if specified
    if return_NDWI:
        out += (scaled_NDWI,)

    # add histogram to the output tuple, if specified
    if return_histogram:
        histogram = cv.calcHist([scaled_NDWI],[0],mask.astype(np.uint8),[256],[0,256])
        out += (histogram,)

    #return the appropriate tuple of results
    return out

def getCoastline(image,mask):
    """
        This function takes a binary label image, and a mask which is being
        applied to the image, and returns the boundary between black and white
        where the mask is not covering it. The boundary is drawn as a line with
        value 255 on a 0-valued background and all masked points being 0 as
        well.

        Arguments:
            *image - a numpy array of the binary label image
            *mask - a boolean numpy array of the mask where parts that should be
                    removed are values 'False'
        Returns:
            *contour_drawing - a binary numpy array of the boundary between
                               black and white from the label
    """
    # add a boundary around the edge, for the line around the edge of the 255-valued area to be drawn, and removed later
    buffered_img = np.pad(image.copy(), ((1,1), (1,1)), 'edge')

    # extract the contours
    contour,hierarchy = cv.findContours(buffered_img[:,:].astype(np.uint8),cv.RETR_LIST,cv.CHAIN_APPROX_NONE)

    # draw the contours onto a new array
    contour_drawing = np.zeros((buffered_img.shape[0],buffered_img.shape[1]),dtype=np.uint8) # prepare the array the coastline will be drawn onto
    cv.drawContours(contour_drawing,contour,-1,(255))
    contour_drawing = contour_drawing[1:-1,1:-1] # remove the buffer added at the start. Now there isn't a border around the edge

    contour_drawing[~mask] = 0 # remove outlines of the masked areas

    return contour_drawing #return the drawn contours

def bufferCoastline(image,mask,kernel_size):
    """
        This function takes a binary label image, and a mask which is being
        applied to the image, and returns a 'coastline buffer', which masks out
        any area a particular distance from the boundary between black and white
        in the label. Buffering is done using a morphological dilation filter
        on the contour drawing from 'getContours'.

        Arguments:
            *image - a numpy array of the binary label image
            *mask - a boolean numpy array of the mask where parts that should be
                    removed are values 'False'
            *kernel_size - the size of the kernel used in the morphological
                           dilation filter
        Returns:
            *coastline_buffer - a boolean numpy array where the area close to
                                the boundary is 'True' and the area away from it
                                is 'False'
    """
    coastline_drawing = getCoastline(image,mask)

    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE,(kernel_size,kernel_size)) # set the kernel form dilation. 10x10 means 100mx100m
    coastline_buffer = cv.dilate(coastline_drawing,kernel,iterations=1)

    return coastline_buffer.astype(np.bool_)

if __name__=='__main__':

    # make a directory to put the outputs into
    out_path = 'CoastlineDetection_Test/'
    if not os.path.exists(out_path):
        os.mkdir(out_path)

    image = np.load('full_image.npy') # load a test image
    DisplayingData.createImage(out_path+'RGB.png',image) # output the image

    mask = np.load('cloud_mask.npy') # load the test mask

    print('Segmenting the coastline...')
    label,thresh,NDWI,hist = detectCoastline(image,mask,return_NDWI=True,return_histogram=True) # coastline detection
    cv.imwrite(out_path+'NDWI.png',NDWI) # output the NDWI
    cv.imwrite(out_path+'label.png',label) # output the label

    # plot the histogram
    scaled_thresh = (thresh+1)*127 # scale the threshold for plotting
    plt.plot(hist) #plot the histogram
    plt.xlim([0,256]) # limit the x-axis
    plt.axvline(scaled_thresh,color='r',linestyle='dashed') #plot the threshold
    plt.text(scaled_thresh*1.05, plt.ylim()[1]*0.9, 'Threshold: {:.2f}'.format(scaled_thresh),color='r') # label the threshold
    plt.xlabel('Intensities') # x-axis label
    plt.ylabel('Count') # y-axis label
    plt.title("Histogram of 'scaled_NDWI'") # title
    plt.savefig(out_path+'histogram.svg') # output an svg vector graphic
    plt.savefig(out_path+'histogram.png') # output a png graphic
    plt.close() # ensures all that is put away again

    print('Buffering the coastline...')
    # make a version of the label, with the mask properly applied to it
    masked_label = label.copy()
    masked_label[~mask] = 0
    cv.imwrite(out_path+'masked_label.png',masked_label) # output the masked label

    # apply a morphological opening filter to the masked label so that small false positives will be missed out of the coastline buffer
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE,(50,50))
    filtered_label = cv.morphologyEx(masked_label, cv.MORPH_OPEN, kernel)
    cv.imwrite(out_path+'filtered_label.png',filtered_label) # output the filtered label

    buffer = bufferCoastline(filtered_label,mask,200) # buffer the coastline
    DisplayingData.createBinaryImage(out_path+'buffer.png',buffer) # output the coastline buffer mask

    print('Applying buffer to the image...')
    DisplayingData.createImage(out_path+'buffered_coastline.png',image,buffer) #output the buffered coastline
