# CSC320 Winter 2017
# Assignment 3
# (c) Olga (Ge Ya) Xu, Kyros Kutulakos
#
# DISTRIBUTION OF THIS CODE ANY FORM (ELECTRONIC OR OTHERWISE,
# AS-IS, MODIFIED OR IN PART), WITHOUT PRIOR WRITTEN AUTHORIZATION
# BY KYROS KUTULAKOS IS STRICTLY PROHIBITED. VIOLATION OF THIS
# POLICY WILL BE CONSIDERED AN ACT OF ACADEMIC DISHONESTY

#
# DO NOT MODIFY THIS FILE ANYWHERE EXCEPT WHERE INDICATED
#

# import basic packages
import numpy as np
import random

# basic numpy configuration

# set random seed
np.random.seed(seed=131)
# ignore division by zero warning
np.seterr(divide='ignore', invalid='ignore')


# This function implements the basic loop of the PatchMatch
# algorithm, as explained in Section 3.2 of the paper.
# The function takes an NNF f as input, performs propagation and random search,
# and returns an updated NNF.
#
# The function takes several input arguments:
#     - source_patches:      The matrix holding the patches of the source image,
#                            as computed by the make_patch_matrix() function. For an
#                            NxM source image and patches of width P, the matrix has
#                            dimensions NxMxCx(P^2) where C is the number of color channels
#                            and P^2 is the total number of pixels in the patch. The
#                            make_patch_matrix() is defined below and is called by the
#                            initialize_algorithm() method of the PatchMatch class. For
#                            your purposes, you may assume that source_patches[i,j,c,:]
#                            gives you the list of intensities for color channel c of
#                            all pixels in the patch centered at pixel [i,j]. Note that patches
#                            that go beyond the image border will contain NaN values for
#                            all patch pixels that fall outside the source image.
#     - target_patches:      The matrix holding the patches of the target image.
#     - f:                   The current nearest-neighbour field
#     - alpha, w:            Algorithm parameters, as explained in Section 3 and Eq.(1)
#     - propagation_enabled: If true, propagation should be performed.
#                            Use this flag for debugging purposes, to see how your
#                            algorithm performs with (or without) this step
#     - random_enabled:      If true, random search should be performed.
#                            Use this flag for debugging purposes, to see how your
#                            algorithm performs with (or without) this step.
#     - odd_iteration:       True if and only if this is an odd-numbered iteration.
#                            As explained in Section 3.2 of the paper, the algorithm
#                            behaves differently in odd and even iterations and this
#                            parameter controls this behavior.
#     - best_D:              And NxM matrix whose element [i,j] is the similarity score between
#                            patch [i,j] in the source and its best-matching patch in the
#                            target. Use this matrix to check if you have found a better
#                            match to [i,j] in the current PatchMatch iteration
#     - global_vars:         (optional) if you want your function to use any global variables,
#                            you can pass them to/from your function using this argument

# Return arguments:
#     - new_f:               The updated NNF
#     - best_D:              The updated similarity scores for the best-matching patches in the
#                            target
#     - global_vars:         (optional) if you want your function to use any global variables,
#                            return them in this argument and they will be stored in the
#                            PatchMatch data structure

###
# source activate open-cv-test
###

###
# source_patches - contains all the patches from the source images in so weird ass format
# source_patches[i,j,c,:] gives you the list of intensities for color channel c of
# all pixels in the patch centered at pixel [i,j]]
###
def propagation_and_random_search(source_patches, target_patches,
                                  f, alpha, w,
                                  propagation_enabled, random_enabled,
                                  odd_iteration, best_D=None,
                                  global_vars=None
                                ):
    new_f = f.copy()

    #############################################
    ###  PLACE YOUR CODE BETWEEN THESE LINES  ###
    #############################################

    ## PROPAGATION
    #print(source_patches.shape)
    #print(f.shape)
    #print(target_patches.shape)

    x_size = source_patches.shape[0]
    y_size = source_patches.shape[1]

    print("sizes", x_size, y_size)

    for i in range(1, source_patches.shape[0] - 1):
        for j in range(1, source_patches.shape[1] - 1):

            #D_original -> D(f(x,y))
            #D_mod_x    -> D(f(x-1, y)) or D(f(x+1, y))
            #D_mod_y    -> D(f(x, y-1)) or D(f(x, y+1))

            v = f[i,j]

            D_original_i, D_original_j = [i,j] + v
            # print(D_original_i, D_original_j)

            #if [i,j] + v is out of range - loop around
            D_original_i, D_original_j = D_original_i %  x_size, D_original_j % y_size

            D_mod_x = None
            D_mod_y = None
            D_original = compute_D(source_patches[i,j], target_patches[D_original_i, D_original_j])

            if(odd_iteration):
                #print("odd")
                odd_v_x = f[i-1, j]
                odd_v_y = f[i, j-1] 

                D_mod_x = compute_D(source_patches[i,j], target_patches[odd_v_x[0] % x_size, odd_v_x[1] % y_size])
                D_mod_y = compute_D(source_patches[i,j], target_patches[odd_v_y[0] % x_size, odd_v_y[1] % y_size])

            else:
               # print("even")
                even_v_x = f[i+1, j] % x_size
                even_v_y = f[i, j+1] % y_size

                D_mod_x = compute_D(source_patches[i,j], target_patches[even_v_x[0] % x_size, even_v_x[1] % y_size])
                D_mod_y = compute_D(source_patches[i,j], target_patches[even_v_y[0] % x_size, even_v_y[1] % y_size])

            #Update f(x,y) with best of the 3
            f[i,j] = min(D_mod_x, D_mod_y, D_original)
            
            ## RANDOM SEARCH
            #alpha - a fixed ratio between search window sizes
            #w - large, max search radius

            k = 0
            while (((alpha ** k) * w ) >= 1):

                #Ri is a uniform random sample from the continuous 2D range.
                R = [np.random.uniform(-1,1), np.random.uniform(-1,1)]

                #u vector from equation 1 in section 3.2 in Barnes paper
                u = f[i,j] + np.multiply(w * (alpha ** k), R)

                k += 1

                #print("R", R, 'w', w, 'alpha', alpha)
           

    #############################################

    return new_f, best_D, global_vars

def compute_D(sourcePatch, destinationPatch):
# Calculation of the D value between vector 1 and 2
# Gonna eventually try all 3 methods

# Method 1:
# CC -> dot product between the vectors
    #print(sourcePatch.shape, destinationPatch.shape)
    #print(np.multiply(sourcePatch, destinationPatch))
    return np.sum(np.multiply(sourcePatch, destinationPatch))
# Method 2:
# NCC -> angle between the vectors

# Method 3:
# RMS -> length of vector joining the 2 vectors together



# This function uses a computed NNF to reconstruct the source image
# using pixels from the target image. The function takes two input
# arguments
#     - target: the target image that was used as input to PatchMatch
#     - f:      the nearest-neighbor field the algorithm computed
# and should return a reconstruction of the source image:
#     - rec_source: an openCV image that has the same shape as the source image
#
# To reconstruct the source, the function copies to pixel (x,y) of the source
# the color of pixel (x,y)+f(x,y) of the target.
#
# The goal of this routine is to demonstrate the quality of the computed NNF f.
# Specifically, if patch (x,y)+f(x,y) in the target image is indeed very similar
# to patch (x,y) in the source, then copying the color of target pixel (x,y)+f(x,y)
# to the source pixel (x,y) should not change the source image appreciably.
# If the NNF is not very high quality, however, the reconstruction of source image
# will not be very good.
#
# You should use matrix/vector operations to avoid looping over pixels,
# as this would be very inefficient

def reconstruct_source_from_target(target, f):
    rec_source = None

    #############################################
    ###  PLACE YOUR CODE BETWEEN THESE LINES  ###
    #############################################


    #############################################

    return rec_source


# This function takes an NxM image with C color channels and a patch size P
# and returns a matrix of size NxMxCxP^2 that contains, for each pixel [i,j] in
# in the image, the pixels in the patch centered at [i,j].
#
# You should study this function very carefully to understand precisely
# how pixel data are organized, and how patches that extend beyond
# the image border are handled.


def make_patch_matrix(im, patch_size):
    phalf = patch_size // 2
    # create an image that is padded with patch_size/2 pixels on all sides
    # whose values are NaN outside the original image
    padded_shape = im.shape[0] + patch_size - 1, im.shape[1] + patch_size - 1, im.shape[2]
    padded_im = np.zeros(padded_shape) * np.NaN
    padded_im[phalf:(im.shape[0] + phalf), phalf:(im.shape[1] + phalf), :] = im

    # Now create the matrix that will hold the vectorized patch of each pixel. If the
    # original image had NxM pixels, this matrix will have NxMx(patch_size*patch_size)
    # pixels
    patch_matrix_shape = im.shape[0], im.shape[1], im.shape[2], patch_size ** 2
    patch_matrix = np.zeros(patch_matrix_shape) * np.NaN
    for i in range(patch_size):
        for j in range(patch_size):
            patch_matrix[:, :, :, i * patch_size + j] = padded_im[i:(i + im.shape[0]), j:(j + im.shape[1]), :]

    return patch_matrix


# Generate a matrix g of size (im_shape[0] x im_shape[1] x 2)
# such that g(y,x) = [y,x]
#
# Step is an optional argument used to create a matrix that is step times
# smaller than the full image in each dimension
#
# Pay attention to this function as it shows how to perform these types
# of operations in a vectorized manner, without resorting to loops


def make_coordinates_matrix(im_shape, step=1):
    """
    Return a matrix of size (im_shape[0] x im_shape[1] x 2) such that g(x,y)=[y,x]
    """
    range_x = np.arange(0, im_shape[1], step)
    range_y = np.arange(0, im_shape[0], step)
    axis_x = np.repeat(range_x[np.newaxis, ...], len(range_y), axis=0)
    axis_y = np.repeat(range_y[..., np.newaxis], len(range_x), axis=1)

    return np.dstack((axis_y, axis_x))
