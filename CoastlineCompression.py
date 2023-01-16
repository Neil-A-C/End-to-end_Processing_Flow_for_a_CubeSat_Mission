import os
import bitarray
import shutil
import cv2 as cv
import sys
import numpy as np
import zipfile
import ast
import DisplayingData
import CoastlineDetection

def encodeCoastline(input,out_zip_name,return_coastline=False):
    """
        This function takes a binary image and outputs a zip file containing
        the contours between black and white, encoded using Freeman 8-direction
        encoding. The output can then be decoded using the 'decodeCoastline'
        function. Outputs a zip, specified by out_zip_name, containing a
        'header.txt' file and 'N' number of contour files named
        'contour0.bin',...,'contourN.bin', where N is the number of contours.

        Arguments:
            *input - a numpy array of the binary image.
            *out_zip_name - the path to the output file
            *return_coastline - When True, the function returns a numpy array of
                                an image with the contours rendered in white on
                                a black background.
    """
    # Create the output directory
    if not os.path.exists(out_zip_name):
        os.mkdir(out_zip_name)

    # Get the shape of the input array
    shape = input.shape

    # Create array of points along the contour
    coastline,hierarchy = cv.findContours(input.astype(np.uint8),cv.RETR_LIST,cv.CHAIN_APPROX_NONE)

    header = np.empty((len(coastline),3),dtype=np.uint16)
    # iterate through the elements in the tuple
    for i in range(len(coastline)):
        contour = coastline[i]
        contour = np.squeeze(contour,axis=1) # squeeze into 2D array
        header[i,0] = contour[0,0]
        header[i,1] = contour[0,1]
        # iterate through the pairs in each array
        symbols = [] # Create array for Freeman symbols
        for j in range(contour.shape[0]-1):
            # compare previous value and current value to encode change with Freeman symbol
            direction = contour[j+1]-contour[j]
            if direction[0] == 1 and direction[1] == 0:
                symbols.append(0)
            elif direction[0] == 1 and direction[1] == 1:
                symbols.append(1)
            elif direction[0] == 0 and direction[1] == 1:
                symbols.append(2)
            elif direction[0] == -1 and direction[1] == 1:
                symbols.append(3)
            elif direction[0] == -1 and direction[1] == 0:
                symbols.append(4)
            elif direction[0] == -1 and direction[1] == -1:
                symbols.append(5)
            elif direction[0] == 0 and direction[1] == -1:
                symbols.append(6)
            elif direction[0] == 1 and direction[1] == -1:
                symbols.append(7)

        # Encode symbols as 3-bit binary numbers
        bin_key = {0:bitarray.bitarray('000'),1:bitarray.bitarray('001'),
                    2:bitarray.bitarray('010'),3:bitarray.bitarray('011'),
                    4:bitarray.bitarray('100'),5:bitarray.bitarray('101'),
                    6:bitarray.bitarray('110'),7:bitarray.bitarray('111')}

        bin = bitarray.bitarray()
        bin.encode(bin_key,symbols)
        if len(bin) == 0:
            bin = bitarray.bitarray('0')

        if len(bin)%8 != 0:
            header[i,2] = 8 - len(bin)%8
        else:
            header[i,2] = 0

        # Store new array in a directory of contours for transmission to ground
        with open(out_zip_name+'/contour'+str(i)+'.bin', "wb") as b:
            bin.tofile(b)

    # Save header file
    with open(out_zip_name+'/header.txt', "w") as h:
        h.write(str(shape)+'\n')
        for i in range(header.shape[0]):
            for j in range(header.shape[1]):
                h.write(str(header[i,j])+'\n')

    # Zip directory
    shutil.make_archive(out_zip_name, 'zip', out_zip_name)

    for i in range(len(coastline)):
        os.remove(out_zip_name+'/contour'+str(i)+'.bin')
    os.remove(out_zip_name+'/header.txt')
    os.rmdir(out_zip_name)

    if return_coastline == True:
        coastline_drawing = np.full(shape,0,dtype=np.uint8) # prepare the array the coastline will be drawn onto
        cv.drawContours(coastline_drawing,coastline,-1,255)
        return coastline_drawing

def decodeCoastline(in_zip_path):
    """
        The function takes decodes contours of a binary image, whcih have been
        encoded using Freeman 8-direction encoding. The input file should be a
        zip, output by the 'encodeCoastline' function, containing a
        'header.txt' file and 'N' number of contour files named
        'contour0.bin',...,'contourN.bin', where N is the number of contours.

        Arguments:
            *in_zip_path - the file path to the input zip file.

        Returns a numpy array of an image with the contours rendered in white on
        a black background.
    """
    with zipfile.ZipFile(in_zip_path+".zip", 'r') as zip_ref:
        zip_ref.extractall(in_zip_path)

    # Read the header
    h = open(in_zip_path+'/header.txt')
    header_list = h.readlines()
    h.close()

    shape = ast.literal_eval(header_list.pop(0)) # use of ast.literal_eval taken from: https://stackoverflow.com/a/16533318
    header = np.empty((len(header_list)//3,3),dtype=np.uint16)
    for i in range(len(header_list)):
        header[i//3,i%3] = header_list[i]

    coastline = []
    for i in range(header.shape[0]): # The number of contours is the number of starting points in the header_list
        # Read the respective contour file
        bin = bitarray.bitarray()
        b = open(in_zip_path+'/contour'+str(i)+'.bin', mode='rb')
        bin.fromfile(b)
        b.close()

        if header[i,2] != 0:
            del bin[-int(header[i,2]):] # Remove buffer bits

        if bin != bitarray.bitarray('0'):

            bin_key = {0:bitarray.bitarray('000'),1:bitarray.bitarray('001'),
                        2:bitarray.bitarray('010'),3:bitarray.bitarray('011'),
                        4:bitarray.bitarray('100'),5:bitarray.bitarray('101'),
                        6:bitarray.bitarray('110'),7:bitarray.bitarray('111')}

            symbols = bin.decode(bin_key)
            contour = np.empty((len(symbols)+1,1,2),dtype = np.int32) # Create an array for the contour to go in
            contour[0] = [header[i,0],header[i,1]]
            direction = [0,0] # Create an array for the directions to be stored in temporarily
            for j in range(len(symbols)):
                if symbols[j] == 0:
                    direction[0] = 1
                    direction[1] = 0
                elif symbols[j] == 1:
                    direction[0] = 1
                    direction[1] = 1
                elif symbols[j] == 2:
                    direction[0] = 0
                    direction[1] = 1
                elif symbols[j] == 3:
                    direction[0] = -1
                    direction[1] = 1
                elif symbols[j] == 4:
                    direction[0] = -1
                    direction[1] = 0
                elif symbols[j] == 5:
                    direction[0] = -1
                    direction[1] = -1
                elif symbols[j] == 6:
                    direction[0] = 0
                    direction[1] = -1
                elif symbols[j] == 7:
                    direction[0] = 1
                    direction[1] = -1

                contour[j+1] = contour[j] + direction

        else:
            contour = np.empty((1,1,2),dtype = np.int32) # Create an array for the contour to go in
            contour[0] = [header[i,0],header[i,1]]

        coastline.append(contour)

    coastline_drawing = np.full(shape,0,dtype=np.uint8) # prepare the array the coastline will be drawn onto
    cv.drawContours(coastline_drawing,coastline,-1,255)

    # delete the unzipped directory
    shutil.rmtree(in_zip_path)

    return coastline_drawing

if __name__=='__main__':

    out_path = "CoastlineCompression_Test/"
    if not os.path.exists(out_path):
        os.mkdir(out_path)

    labels = ["simple_coast","river","delta","sand_bar"]

    mask = np.ones((256,256),dtype=np.bool_)
    mask[100:-100,100:-100] = False
    mask[10:30,10:30] = False
    mask[-40:-10,-50:-30] = False

    # Output a visible version of the mask
    visible_mask = np.zeros(mask.shape,dtype=np.uint8)
    visible_mask[mask] = 255
    cv.imwrite(out_path+'mask.png',visible_mask)

    for label_name in labels:
        print(label_name+':',end=' ')
        #load the input image
        label = np.load(label_name+".npy")

        label = np.transpose(label,(1,2,0)) # Transpose the image so that the dimensions are 256x256x1
        label = np.squeeze(label) #squeeze the array down to 256x256 dimensions

        directory = out_path+label_name+'/' # Set the naming scheme for output zip and image files
        if not os.path.exists(directory):
            os.mkdir(directory)

        # Mask the label
        label[label==1] = 255
        label[~mask] = 127
        cv.imwrite(directory+'masked_label.png',label)

        coastline_drawing = CoastlineDetection.getCoastline(label,mask) # get the coastline, with the mask applied

        # encode the coastline
        encoded_coastline = encodeCoastline(coastline_drawing,directory+'compressed',return_coastline=True)
        decoded_coastline = decodeCoastline(directory+'compressed')

        # output the coastline that was encoded and decoded
        cv.imwrite(directory+"encoded.png",encoded_coastline)
        cv.imwrite(directory+"decoded.png",decoded_coastline)

        # check that the decoded coastline matches the original, implying lossless transmission
        if (encoded_coastline == decoded_coastline).all():
            print("Match!")
        else:
            print("No Match!")
