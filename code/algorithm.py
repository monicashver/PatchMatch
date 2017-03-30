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
import time
np.set_printoptions(threshold=np.inf)
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

    print(random_enabled, propagation_enabled)
    x_size, y_size = source_patches.shape[0], source_patches.shape[1]
    old_D = best_D
    #initialize best_D to a matrix of Nan
    if(best_D is None):
        best_D = np.empty((x_size, y_size)) * np.nan

    if(odd_iteration):
        start_x = 1
        start_y = 1
        end_x = source_patches.shape[0] - 2
        end_y = source_patches.shape[1] - 2
        offset = 1
        loop = 1
    else:
        start_x = source_patches.shape[0] - 2
        start_y = source_patches.shape[1] - 2
        end_x = 0#source_patches.shape[0] - 1
        end_y = 0#source_patches.shape[1] - 1
        offset = 1
        loop = -1
    #print(start_x, start_y, end_x, end_y, offset, loop)
    #print("START", start, "X", end_x, "Y", end_y, "ODD", odd_iteration)
    k = int(np.ceil(- np.log10(w)/ np.log10(alpha)))

    same_count = 0
    for i in range(start_x, end_x, loop):
        for j in range(start_y, end_y, loop):
            #print(i,j)

            ##add check for if we can checks the offsets or not
            #check
            #D_original -> D(f(x,y))
            #D_mod_x    -> D(f(x-1, y)) or D(f(x+1, y))
            #D_mod_y    -> D(f(x, y-1)) or D(f(x, y+1))

            #f[i,j]-> [x,y]

            if(not propagation_enabled):
                D_original_i, D_original_j = i + f[i,j,0],  j + f[i,j,1]

                #if [i,j] + v is out of range
                D_original_i = np.clip(D_original_i, -x_size, x_size-1)
                D_original_j = np.clip(D_original_j, -y_size, y_size-1)

                D_mod_horizontal = None
                D_mod_verticle = None
                D_original = compute_D(source_patches[i,j], target_patches[D_original_i, D_original_j])
                #print("\n")

                #print("ORIGINAL", f[i,j], "val", D_original)
                #print(D_original)
                v_horizontal = f[i+offset, j]
                v_verticle   = f[i, j+offset]
                #print("\n") 
                if(v_horizontal[0] + i < x_size and v_horizontal[1] + j < y_size):
                    #print("validH")
                    # x_target = np.clip(i + v_horizontal[0], -x_size, x_size-1)
                    # y_target = np.clip(i + v_horizontal[1], -y_size, y_size-1)

                    x_target = i + v_horizontal[0]  
                    y_target = j + v_horizontal[1]
                    #print(i, j)
                    #print("X target", x_target)
                    #print("Y target", y_target)
                    D_mod_horizontal = compute_D(source_patches[i,j], target_patches[x_target, y_target])
                    #print("HORIZONAL", v_horizontal, "val", D_mod_horizontal)
                else:
                    D_mod_horizontal = np.nan
                if(v_verticle[0] + i < x_size and v_verticle[1] + j < y_size):
                    #print("validV")
                    #v_verticle   = f[i, j+offset] 

                    # x_target = np.clip(i + v_verticle[0], -x_size, x_size-1)
                    # y_target = np.clip(i + v_verticle[1], -y_size, y_size-1)

                    x_target = i + v_verticle[0]
                    y_target = j + v_verticle[1]

                    #print(i, j)
                    #print("X target", x_target)
                    #print("Y target", y_target)
                    D_mod_verticle = compute_D(source_patches[i,j], target_patches[x_target, y_target])
                    #print("VERTICLE",  v_verticle, "val", D_mod_verticle)
                else:
                    D_mod_verticle = np.nan

                # Update best_D accordingly
                # check if offset values are ok
                D = [D_mod_horizontal, D_mod_verticle, D_original]
                F_s = [v_horizontal, v_verticle, f[i,j]]
                min_val = np.nanmin(D) 
                #all min values are nan

                if(np.isnan(min_val)):
                    continue

                index = D.index(min_val)

                #if best_D is nan -> update with it minimum calculated value
                if(best_D[i,j] == min_val):
                    #print("SAME", best_D[i,j], D, index)
                    same_count += 1
                if(np.isnan(best_D[i,j])):
                    #print("UPDATED", best_D[i,j], min_val)
                    best_D[i, j] = min_val
                    f[i,j] = F_s[index]
                    new_f[i,j] = F_s[index]

                elif(min_val < best_D[i,j]):
                    #print("UPDATED", best_D[i,j], min_val)
                    best_D[i, j] = min_val
                    f[i,j] = F_s[index]
                    new_f[i,j] = F_s[index]
               #else:


            ## RANDOM SEARCH ##
            if(not random_enabled):
                for l in range(k):

                    #Ri is a uniform random sample from the continuous 2D range.
                    R = [np.random.uniform(-1,1), np.random.uniform(-1,1)]

                    #u vector from equation 1 in section 3.2 in Barnes paper
                    u = f[i,j] + np.multiply((alpha ** l) * w, R)
                    
                    #new offset values
                    x = i + u[0]
                    y = j + u[1]

                    #make sure x and y values are valid
                    x = int(np.clip(x, -x_size, x_size-1))
                    y = int(np.clip(y, -y_size, y_size-1))

                    new_dist = compute_D(source_patches[i,j], target_patches[x, y])

                    # if we found a closer value -> update f and original D
                    if(new_dist < D_original):
                        new_f[i,j] = u
                        #f[i,j] = u
                        D_original = new_dist

    #############################################
    #print(best_D == old_D)
    print(same_count)
    return new_f, best_D, global_vars

def compute_D(source_patch, destination_patch):
# Calculation of the D value between vector 1 and 2
    temp1 = np.ndarray.flatten(source_patch)
    temp2 = np.ndarray.flatten(destination_patch)

    dist = np.abs(temp1 - temp2)

    return dist.dot(dist)


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

    rec_source = np.zeros(target.shape)

    for i in range(target.shape[0]):
        for j in range(target.shape[1]):

            x = (i + f[i,j,0])
            y = (j + f[i,j,1]) 

            rec_source[i,j] = target[x, y]

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
