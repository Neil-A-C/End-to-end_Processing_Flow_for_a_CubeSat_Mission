import os
import numpy as np
import cv2 as cv

# libraries of each sub-system
import DisplayingData
import CloudMasking
import CoastlineDetection
import CoastlineCompression
import ImageCompression

# make a directory to put the outputs into
out_path = 'Full_Test/'
if not os.path.exists(out_path):
    os.mkdir(out_path)

safe_dir = 'S2A_MSIL1C_20220826T115411_N0400_R023_T30VUK_20220826T171057.SAFE' # name of the test input .SAFE directory

# ***Extractng Full Image***
print('Extracting Full Image...')
# extract the image from the .SAFE directory
image = DisplayingData.extractBands(safe_dir,(5490,5490)) # output an RGB image of the data
np.save(out_path+'full_image.npy',image)
DisplayingData.createImage(out_path+'RGB.png',image)
print('\n')


# ***Cloud Masking***
print('Cloud Masking...')
cloud_classification_file = out_path+'cloud_classification.tif' # specify the name of the classification file

CloudMasking.cloudClassification(safe_dir,cloud_classification_file) # run FMask classification

# create a mask from the FMask classification
cloud_mask = CloudMasking.createMask(cloud_classification_file)
DisplayingData.createBinaryImage(out_path+'cloud_mask.png',cloud_mask)

# output the masked image
DisplayingData.createImage(out_path+'cloud_masked_RGB.png',image,cloud_mask)
print('\n')

# ***Coastline Detection***
print('Coastline Detection...')
print('Segmenting the coastline...')
label,thresh,NDWI = CoastlineDetection.detectCoastline(image,cloud_mask,return_NDWI=True) # coastline detection
cv.imwrite(out_path+'NDWI.png',NDWI) # output the NDWI
cv.imwrite(out_path+'label.png',label) # output the label

print('Buffering the coastline...')
# make a version of the label, with the mask properly applied to it
masked_label = label.copy()
masked_label[~cloud_mask] = 0
cv.imwrite(out_path+'masked_label.png',masked_label) # output the masked label

# apply a morphological opening filter to the masked label so that small false positives will be missed out of the coastline buffer
kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE,(50,50))
filtered_label = cv.morphologyEx(masked_label, cv.MORPH_OPEN, kernel)
cv.imwrite(out_path+'filtered_label.png',filtered_label) # output the filtered label

coastline_buffer = CoastlineDetection.bufferCoastline(filtered_label,cloud_mask,200) # buffer the coastline
coastline_mask = np.bitwise_and(coastline_buffer,cloud_mask) # Mask the label with the coastline buffer and cloud mask
coastline_buffer = CoastlineDetection.bufferCoastline(label,coastline_mask,200) # buffer the coastline

print('Applying buffer to the image...')
DisplayingData.createImage(out_path+'buffered_coastline.png',image,coastline_buffer) #output the buffered coastline
print('\n')


# *** Coastline Compression***
print('Coastline Compression...')
# Output a visible version of the coastline mask
visible_coastline_mask = np.zeros(cloud_mask.shape,dtype=np.uint8)
visible_coastline_mask[coastline_mask] = 255

# Mask the label
label[label==1] = 255
label[~coastline_mask] = 255

coastline_drawing = CoastlineDetection.getCoastline(label,coastline_mask) # get the coastline, with the mask applied
cv.imwrite(out_path+'coastline.png',coastline_drawing)

# encode the coastline
encoded_coastline = CoastlineCompression.encodeCoastline(coastline_drawing,out_path+'compressed_coastline',return_coastline=True)
decoded_coastline = CoastlineCompression.decodeCoastline(out_path+'compressed_coastline')

# output the decoded coastline
cv.imwrite(out_path+"decoded_coastline.png",decoded_coastline)

# check that the decoded coastline matches the original, implying lossless transmission
if (encoded_coastline == decoded_coastline).all():
    print("Match!")
else:
    print("No Match!")

print('\n')

# ***Image Compression***
print('Image Compression...')
# mask the full image
full_mask = np.dstack([coastline_buffer]*image.shape[2])
buffered_full_image = image.copy()
buffered_full_image[~full_mask] = 0

# compress the buffered full image
print('Compression:')
ImageCompression.multibandCompression(out_path+'compressed_image',buffered_full_image,3)
print('Complete!')

# decompress the buffered full image
print('Decompression:')
decompressed_full_image = ImageCompression.multibandDecompression(out_path+'compressed_image',3)
# checks that the decompressed image is identical to the input, implying lossless compression
if (decompressed_full_image == buffered_full_image).all():
    print('Success!')
else:
    print('Failed!')

# calculate the compression Ratio
in_size = os.path.getsize(out_path+'full_image.npy') # get the size of the raw input image
compressed_size = os.path.getsize(out_path+'compressed_image.zip') # get the size of the compressed image
print('Compression Ratio:',in_size/compressed_size)

# output the decompressed image
DisplayingData.createImage(out_path+'decompressed_image.png',decompressed_full_image)
