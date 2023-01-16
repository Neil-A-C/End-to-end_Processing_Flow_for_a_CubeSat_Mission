import math
import os
import shutil
import zipfile
import numpy as np
import cv2 as cv
import bitarray
import bitarray.util
from skimage import data
from time import process_time
import DisplayingData

def recursiveSplit(array_list):
    """
        This recursive function splits up the input array into quadrants and
        records their maximum values. The maximum values are recorded relative
        to the parent quadrant to assist the option of compressing the L_max
        data stream in future.

        It then checks if all values in the quadrants are
        equal. If they are, the 'branch' from that 'node' is recorded as a
        'leaf' with a 0 in the bitarray T. If they,re not, it splits them
        further and repeats the process on the next level down the data tree.

        Finally, T and L_max are returned.
    """
    T = bitarray.bitarray() # Initialise T
    L_max = [] # Initialise L_max
    nodes = len(array_list) # Find the number of nodes at this level
    split = [] # An array that all of the split arrays will be put into
    for i in range(nodes):
        array = array_list[i] # get the array for each node
        max = np.amax(array) # gets the max value in this array
        #split the array into four quadrants
        split_1_2 = np.hsplit(np.vsplit(array,2)[0],2)
        split_3_4 = np.hsplit(np.vsplit(array,2)[1],2)
        quadrants = (split_1_2[0],split_1_2[1],split_3_4[0],split_3_4[1])

        for quadrant in quadrants: # for each of the four quadrants
            L_max.append(max - np.amax(quadrant)) # add the relative max value to L_max list
            if np.shape(quadrant)[0] != 1: # this is not the bottom of the tree
                if np.any(quadrant != quadrant[0,0]): # notes that this branch is another node
                    T.append(1)
                    split.append(quadrant)
                else:
                    T.append(0) # notes that this is a mid-level leaf
            else: # If possible, the bottom level of T could be removed, so long as L_max can be used instead to navigate this
                T.append(0) # notes that this is the bottom of the tree

    if len(split) != 0: # recur if there is a further level of nodes below this one
        add_T,add_L_max = recursiveSplit(split)
        T.extend(add_T)
        L_max.extend(add_L_max)

    return T,L_max # return the completed T and L_max data

def recursiveBuild(T,L_max,level,shape):
    """
        This function takes data from a k2-raster compression and re-builds the
        image.
    """
    level_T,level_max = level # unpack the data on this level of the data tree
    quad_shape = shape//2 # the shape of the quadrants is half the shape of the full array
    nodes = [] # initialise a list of the children which are nodes
    quadrants = [] # initialise a list to put all the quadrants into
    next_level_T = bitarray.bitarray() # initialise a bitarray to put the next level's T data into
    next_level_max = [] # initialise a bitarray to put the next level's L_max data into

    for i in range(len(level_T)): # for all branches off all nodes at this level
        if level_T[i] == 1: # if the branch is another node
            # move the four values in T and L_max into the next level data
            next_level_T.extend(T[:4])
            next_level_max.extend([level_max[i]-x for x in L_max[:4]]) # calculates the actual max value, based on the max value of the parent node and differenc stored in L_max
            del T[:4]
            del L_max[:4]
            nodes.append(0) # put a 0 in the nodes list to show that this still needs filled with an array made up of quadrants from the next level down
        else: # if this is a leaf, fill a quadrants with the max value and put it in the nodes list
            nodes.append(np.full((quad_shape,quad_shape),level_max[i]))

    if len(next_level_T) != 0:  # if there is another level beyond this one, recur the function with the next level data we have collected
        next_level = next_level_T,next_level_max
        lower_branches = recursiveBuild(T,L_max,next_level,quad_shape) # recursion returns a list of branches

    # go back through the nodes list, filling in any indices that were marked as nodes with their correct quadrants, as found in recursion of this function
    for i in range(len(nodes)):
        if len(next_level_T) != 0:
            if type(nodes[i]) == int:
                nodes[i] = lower_branches[0]
                del lower_branches[0]
        # after each group of four nodes has been cycled through, they are joined together and put in the quadrants array
        if i % 4 == 3:
            split_1_2 = np.hstack((nodes[i-3],nodes[i-2]))
            split_3_4 = np.hstack((nodes[i-1],nodes[i]))
            quadrants.append(np.vstack((split_1_2,split_3_4)))

    return quadrants # returns a list of all the quadrants at this level that need to be put in to 'node[i]==0 spaces' on the level above

def buffer(array):
    """
        This function pads an array so that it becomes a square with sides of
        2^n, where n is a positive integer.

        This is the required format for the k2-raster image compression
        algorithm.

        Uses numpy.pad with the 'edge' padding mode so as to maximise the
        possibility of higher-level quadrants in the k2-raster being all the
        same value, thus increasing the compression ratio.

        Arguments:
            *array - a numpy array with at least two dimensions
        Returns:
            *buffered_array - the same array, padded to the correct shape.
            *buffered_dim - the dimensions of the buffered array
    """
    dim = array.shape

    if dim[0] < dim[1]: # if x-axis is shorter than y-axis
        buffered_dim = dim[1]
    else:
        buffered_dim = dim[0]

    log_dim = math.log(buffered_dim,2) # log_2 of the dimensions of the array
    if log_dim != int(log_dim): # tests whether log_2 of the dimensions is an integer
        buffered_dim = 2**(int(log_dim)+1) # if not, set new taget dimensions to make that the case

    # Pad the array
    y_pad = buffered_dim - dim[0] # add padding to the y_axis
    x_pad = buffered_dim - dim[1] # add padding to the x_axis
    padding = ((0,y_pad),(0,x_pad)) # set up the 'padding' argument for np.pad
    buffered_array = np.pad(array,padding,mode='constant') # uses 'edge' to maximise quadrants in k2-raster with all same values

    return buffered_array,buffered_dim

def DACsEncode(list,S):
    """
        This function carries out DACs variable-length integer encoding.

        Each integer in the input list is converted to it's binary form and
        broken into chunks of S number of bits. The first chunk has a 0 put at
        the front and the follwoing chunks are preceded with 1s so that the
        decoder will no when a new integeer begins. All of these encoded
        integers can then be put into a single continuous binary array.

        Arguments:
            *list - a list of integers
            *S - the number of bits to be stored in each chunk
        Returns:
            *encoded - a bitarray of all the encoded integers
    """
    encoded = bitarray.bitarray()
    for i in range(len(list)):
        # convert the value to a bitarray
        binary = bin(list[i])
        binary = binary[2:]
        bit_array = bitarray.bitarray(binary)

        # pad the front if first chunk will not be full
        pad = len(bit_array)%S
        if pad != 0:
            bit_array.reverse()
            for i in range(S-pad):
                bit_array.extend('0')
            bit_array.reverse()

        chunk = bitarray.bitarray() # initialise a bitarray for the chunk
        while len(bit_array) != 0:
            if len(chunk) == 0:
                chunk.extend('0') # first bit is zero if this is the first chunk of a new integer
            else:
                chunk.extend('1') # for following chunks of the same integer, the first bit is 1
            chunk.extend(bit_array[:S]) # fill the rest of the chunk with info about the number
            del bit_array[:S]

        encoded.extend(chunk)

    return encoded

def DACsDecode(bit_array,S):
    """
        This function decodes a DACs encoded list of integers.

        Arguments:
            *bit_array - a bitarray of DACs encoded integers, like that returned
                         by DACsEncode
            *S - the size of chunks that the list is encoded into
        Returns:
            *decoded - a list of integers
    """
    bin_list = [] # initialise the list of binary integers
    chunk_length = S+1 # calculate the length of each chunk, based on S
    for i in range(len(bit_array)//(S+1)):
        chunk = bit_array[(chunk_length*i):(chunk_length*i+chunk_length)] # read a chunk
        if chunk[0] == 0:
            bin_list.append(chunk[1:]) # start a new integer, if the chunk starts with 0
        else:
            bin_list[-1].extend(chunk[1:]) # otherwise, continue the same integer

    decoded = [] # initialise the output list
    # convert the binary integers to python integers
    for i in range(len(bin_list)):
        decoded.append(bitarray.util.ba2int(bin_list[i]))

    return decoded

def compress(array,DACs_S):
    """
        This function encodes a 2-dimensional numpy array using the k2-raster
        method.

        It uses the 'buffer' function to make sure that the array is the right
        shape for k2-raster compression then calls the recursiveSplit function
        to get the T and L_max encoded data. Any additional useful information
        is then added to the beginning of L_max before it is encoded further
        using DACs variable-length integer encoding.

        Arguments:
            *array - the array that will be encoded
            *DACs_S - the 'S' value which will be used in DACs encoding of L_max
        Returns:
            *T - a bitarray describing the shape of the k2-raster tree
            *L_bin - a bitarray of the DACs encoded list of integerscontaing
                     information about the dimensions of the array and values
                     within it
    """
    dim = array.shape
    # buffer the array
    array,buffered_dim = buffer(array)
    buffered_dim = int(math.log(buffered_dim,2)) # the buffered dimensions of the image are encoded by log_2 for a smaller integer.

    # Get the K2Raster compressed data
    r_max = np.amax(array)
    r_min = np.amin(array)
    T,L_max = recursiveSplit([array])

    # make up the L_max list that will be encoded
    L = [r_max,dim[0],dim[1],buffered_dim]
    L.extend(L_max)

    L_bin = DACsEncode(L,DACs_S)

    return T,L_bin

def decompress(T,L,DACs_S):
    """
        This function decompresses data which has been compressed using the
        k2-raster method.

        It uses the 'DACsDecode' function to decode the L_max integer list,
        which then has the information about the dimensions of the array read
        off the start. Then 'recursiveBuild' is called to build up the array
        from the available information.

        Arguments:
            *T - the 'T' bitarray encoded by 'compress'
            *L - the 'L_bin' bitarray encoded by 'compress'
            *DACs_S - the chunk sizes used in DACs encoding in 'compress', which
                      will be passed to DACsDecode
        Returns:
            *out_arr - a 2-dimensional numpy array, which will be identical to
                       the input to compress, when this data was encoded
    """
    L_max = DACsDecode(L,DACs_S)

    # Get r_max_in from first entry in L_max
    r_max = L_max[0]
    del L_max[0]

    # Get dimensions from L_max_in
    dim = (L_max[0],L_max[1])
    del L_max[:2]
    # dim_in = (2**dim_in[0],2**dim_in[1])

    # Get buffered dimensions from L_max_in
    buffered_dim = L_max[0]
    del L_max[0]
    buffered_dim = 2**buffered_dim

    #get the first level of T and L_max ready
    level = (T[:4],([r_max-x for x in L_max[:4]]))
    del T[:4]
    del L_max[:4]

    # build the image back up from the compressed data
    out_arr = np.asarray(recursiveBuild(T,L_max,level,buffered_dim)[0])

    # unbuffer the image
    return out_arr[:dim[0],:dim[1]]

def multibandCompression(out_file,image,S):
    """
        This function carries out k2-raster compression of a multispectral image
        by iterating through the bands of a 3-dimensional numpy array and
        passing each 2-dimensional band to the 'compress' function. Each band's
        data is then stored seperately in two files for that numbered band.

        At the end, all of the data is put into a .zip file.

        Arguments:
            *out_file - the file path for the output zip file
            *image - a 3-dimensional numpy array of an image which will be
                     compressed
            *S - the size of chunks which will be used in DACs encoding of L_max
    """
    # Create target directory
    if not os.path.exists(out_file+'/'):
        os.mkdir(out_file+'/')

    for i in range(image.shape[2]): # iterate through all the bands of the image

        print('Compressing band',str(i+1))

        T,L_bin = compress(image[:,:,i],S)

        # Save T
        with open(out_file+'/band_'+str(i)+'_T.bin','wb') as t:
            T.tofile(t)

        # Save L_max
        with open(out_file+'/band_'+str(i)+'_L_max.bin','wb') as l:
            L_bin.tofile(l)

    # Zip directory
    shutil.make_archive(out_file,'zip',out_file)

    # delete the directory that was just zipped together
    shutil.rmtree(out_file)

def multibandDecompression(in_file,S):
    """
        This function decompresses a 3-dimensional array which has been
        compressed by 'multibandCompression'.

        It infers the number of bands from the number of files in the input
        zip file then iterates through each band, decompressing each in turn
        using the 'decompress' function. the resulting list of bands is finally
        combined into a single 3-dimensional numpy array which will be identical
        to the input array to 'multibandCompression' when the image was
        compressed.

        Arguments:
            *in_file - the file path to the .zip file of the compressed data
        Returns:
            *out_arr - the decompressed 3-dimensional numpy array
            *S - the size of chunks which were used in DACs encoding of L_max
    """
    # Read the zip
    with zipfile.ZipFile(in_file+'.zip','r') as zip_ref:
        zip_ref.extractall(in_file)

    #the number of bands in the image is half the number of files in the directory
    no_of_files = len(os.listdir(in_file))
    no_of_bands = no_of_files//2

    out_arrs = [] # initialise a list of all the band arrays
    for i in range(no_of_bands): # iterate through the bands
        print('Decompressing band',str(i+1))
        # Read the T file
        T_in = bitarray.bitarray()
        with open(in_file+'/band_'+str(i)+'_T.bin',mode='rb') as t:
            T_in.fromfile(t)

        # Read the L_max file
        L_in = bitarray.bitarray()
        with open(in_file+'/band_'+str(i)+'_L_max.bin',mode='rb') as l:
            L_in.fromfile(l)

        out_arrs.append(decompress(T_in,L_in,S)) # decompress the band and add it to the list of arrays

    # make up the full image from all the bands
    out_arr_shape = (out_arrs[0].shape[0],out_arrs[0].shape[1],no_of_bands) # determine the shape from the 2 dimensions of the bands and the number of bands inferred from the number of files
    out_arr = np.empty(out_arr_shape,dtype=np.uint8) # initialise an output array
    # fill the array with the bands
    for i in range(len(out_arrs)):
        out_arr[:,:,i] = out_arrs[i]

    # delete the unzipped directory
    shutil.rmtree(in_file)

    return out_arr


if __name__=='__main__':

    # make a directory to put the outputs into
    out_path = 'ImageCompression_Test/'
    if not os.path.exists(out_path):
        os.mkdir(out_path)

    zip_name = out_path + 'out'

    # A test array
    print('Extracting image data')
    # uses the buffered coastline output by the CoastlineDetection test script
    test_image = cv.imread('CoastlineDetection_Test/buffered_coastline.png')
    S = 3

    cv.imwrite(out_path+'in_image.bmp',test_image) # output the raw image

    print('Compression:')
    CT_start = process_time() # start compression timer
    multibandCompression(zip_name,test_image,S)
    CT_end = process_time() # end compression timer
    print('Compression Complete!')
    print()

    print('Decompression:')
    DT_start = process_time() # start decompression timer
    out_arr = multibandDecompression(zip_name,S)
    cv.imwrite(out_path+'out_image.png',out_arr)
    DT_end = process_time() # end decompression timer
    print('Decompression Complete!')
    print()

    # checks that the decompressed image is identical to the input, implying lossless compression
    if (out_arr == test_image).all():
        print('Success!')
    print()

    # calculate compression ratio
    in_size = os.path.getsize(out_path+'in_image.bmp') # get the size of the raw input image
    compressed_size = os.path.getsize(zip_name+'.zip') # get the size of the compressed image
    CR = in_size/compressed_size # calculate the compression ratio
    CT = CT_end - CT_start # calculate the compression time
    DT = DT_end - DT_start # calculate the decompression time

    # print the results
    print('Compression Ratio: {:.2f}'.format(CR))
    print('Compression Time: {:.2f}s'.format(CT))
    print('Decompression Time: {:.2f}s'.format(DT))
