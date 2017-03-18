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

# import packages
from nnf import *
from algorithm import *
import cv2 as cv
import numpy as np
import time

import os

# A decorator function for elapsed-time profiling
def profile(fn):
    def with_profiling(*args, **kwargs):
        start_time = time.time()

        ret = fn(*args, **kwargs)
        elapsed_time = time.time() - start_time

        print '\tFunction {} ran {:.4f}'.format(fn.__name__, elapsed_time)

        return ret

    return with_profiling


#####################################################################
#
# The PatchMatch Class
#
# This class contains the basic methods required for implementing
# the PatchMatch algorithm. Description of the individual methods is
# given below.
#
# To run PatchMatch one must create an instance of this class. See
# function run() in file run.py for an example of how it is called
#
#####################################################################


class PatchMatch:
    #
    # The class constructor
    #
    # When called, it creates a private dictionary object that acts
    # as a container for all input and all output images of
    # the inpainting algorithm. These images are initialized to None
    # and populated/accessed by calling the the readImage(), writeImage(),
    # and run_iterations() methods.
    #
    def __init__(self):
        self._images = {
            'source': None,
            'target': None,
            'NNF-image': None,
            'NNF-vectors': None,
            'rec-source': None,
        }
        # set default parameters
        self._iters = None
        self._patch_size = None
        self._alpha = None
        self._w = None
        self._im_shape = None
        self._f = None
        self._best_D = None
        self._disable_random = None
        self._disable_propagation = None
        self._output = None
        self._partial_results = None
        self._NNF_vectors = None
        self._NNF_image = None
        self._rec_source = None
        self._server = None
        self._NNF_subsampling = None
        self._NNF_line_width = None
        self._NNF_line_color = None
        self._tmpdir = None
        # internal algorithm variables
        self._need_init = True
        self._source_patches = None
        self._target_patches = None
        self._current_iteration = None
        self._init_NNF_filename = None
        self._global_vars = None


    # Use OpenCV to read an image from a file and copy its contents to the
    # PatchMatch instance's private dictionary object. The key
    # specifies the image variable and should be one of the
    # strings in lines 70-75.
    #
    # The routine should return True if it succeeded. If it did not, it should
    # leave the matting instance's dictionary entry unaffected and return
    # False, along with an error message
    def read_image(self, filename, key):
        success = False
        msg = 'No Image Available'

        #########################################
        ## PLACE YOUR CODE BETWEEN THESE LINES ##
        #########################################

        # COPY INTO THIS SPACE YOUR IMPLEMENTATION OF THIS FUNCTION
        # FROM YOUR algorithm.py of A1

        #Check that valid key given
        if (not key in self._images):
            success, msg = False, 'Invalid key provided: ' + key
            
        #Check if given filename is valid
        elif (not os.path.isfile(filename)):
            success, msg = False, 'Invalid filename provided: ' + filename
        else:
            #Attemp to load image
            picture = cv.imread(filename, cv.IMREAD_UNCHANGED)
            
            #imread failed
            if (type(picture) == None):
                success, msg = False, 'Failed to load image properly'
                
            #imread was successful 
            else:
                self._images[key] = picture
                success, msg = True, "Successfully loaded image"      

        #########################################
        return success, msg

    # Use OpenCV to write to a file an image that is contained in the
    # instance's private dictionary. The key specifies the which image
    # should be written and should be one of the strings in lines 89-95.
    #
    # The routine should return True if it succeeded. If it did not, it should
    # return False, along with an error message
    def write_image(self, filename, key):
        success = False
        msg = 'No Image Available'

        #########################################
        ## PLACE YOUR CODE BETWEEN THESE LINES ##
        #########################################

        # COPY INTO THIS SPACE YOUR IMPLEMENTATION OF THIS FUNCTION
        # FROM YOUR algorithm.py of A1
        
        data = self._images[key]
        
        #Invalid key provided
        if(key not in self._images):
            success, msg = False, 'Invalid key provided'
        
        #Valid key, but value for that key is None
        elif (type(data) == None):
            success, msg = False, 'There is no data in key: ' + key + 'to write'
        else:
            cv.imwrite(filename, data)
            success = True, 'Successfully wrote image'

        #########################################
        return success, msg

    #
    # Reconstruct the source image using pixels from the target.
    # See algorithm.py for details. You will need to complete the function
    # in that file.
    #
    @profile
    def _reconstruct_source(self):
        """
        Reconstruct the source image using the target image and the current nearest neighbour field.
        """
        self._images['rec-source'] = \
            reconstruct_source_from_target(self._images['target'], self._f)

    #
    # Implement one iteration of the PatchMatch algorithm
    # See algorithm.py for details. You will need to complete the function
    # in that file.
    #
    @profile
    def _propagation_and_random_search(self):
        """
        Implement the propagation and random search steps of the PatchMatch algorithm.
        """
        odd_iter = self._current_iteration % 2 != 0


        self._f, self._best_D, self._global_vars = propagation_and_random_search(self._source_patches,
                                                                self._target_patches,
                                                                self._f,
                                                                self._alpha, self._w,
                                                                self._propagation_enabled,
                                                                self._random_enabled,
                                                                odd_iter,
                                                                self._best_D,
                                                                self._global_vars
                                                              )

    #
    # Initialize the variables required for PatchMatch
    #

    def initialize_algorithm(self):
        if self._images['source'] is not None:
            self.set_im_shape()
            self._source_patches = make_patch_matrix(self._images['source'], self._patch_size)
        else:
            self._source_patches = None
        if self._images['target'] is not None:
            self._target_patches = make_patch_matrix(self._images['target'], self._patch_size)
        else:
            self._target_patches = None
        if self._w == 0:
            # if the maximum search radius was not specified, we use the
            # maximum image dimension of the source image
            self._w = np.max(self._images['source'].shape[0:2])
        self._current_iteration = 1
        self._best_D = None
        self._need_init = False

    #
    # Execute one iteration of the PatchMatch algorithm
    #

    def _validate(self):
        return ((self._images['source'] is not None)
            and (self._images['target'] is not None)
            and (self._source_patches is not None)
            and (self._target_patches is not None)
            and (self._f is not None)
            and (self._patch_size > 0)
            and (self._images['source'].shape[0] == self._images['target'].shape[0])
            and (self._images['source'].shape[1] == self._images['target'].shape[1])
            and (self._f.shape[0] == self._images['source'].shape[0])
            and (self._f.shape[1] == self._images['source'].shape[1]))

    def step_algorithm(self):
        # initialize the algorithm data structures if this is the first run
        if self._need_init:
            self.initialize_algorithm()
        success = False
        # make sure all the data we need are available
        if self._validate():
            if self._current_iteration <= self._iters:
                print 'Running iteration {}...'.format(self._current_iteration)
                self._propagation_and_random_search()
                self._current_iteration += 1
                success = True
        else:
            return success
        if (self._current_iteration > self._iters) or self.partial_results():
            # write the output files
            if self.NNF_image():
                self._images['NNF-image'] = create_NNF_image(self._f)
                ok, msg = self.write_image(self.make_filename('nnf-col','png', not success), 'NNF-image')
                if not ok:
                    print 'Error: write_image: ', msg

            if self.NNF_vectors():
                # this is a kludge: the need to use matplotlib to write the
                # image to a file, then we re-read it into an openCV image,
                # then finally write that openCV image into the desired file
                ok, msg = self.read_image(
                    create_NNF_vectors_image(self._images['source'],
                                             self._images['target'],
                                             self._f,
                                             self._patch_size,
                                             subsampling=self._NNF_subsampling,
                                             line_width=self._NNF_line_width,
                                             line_color=self._NNF_line_color,
                                             tmpdir=self._tmpdir),
                    'NNF-vectors')
                ok, msg = self.write_image(self.make_filename('nnf-vec','png', not success), 'NNF-vectors')
                if not ok:
                    print 'Error: write_image: ', msg
            if self.rec_source():
                self._reconstruct_source()
                ok, msg = self.write_image(self.make_filename('rec-src','png', not success), 'rec-source')
                if not ok:
                    print 'Error: write_image: ', msg
            ok, msg = save_NNF(self.NNF(), self.make_filename('nnf','npy', not success),)
            if not ok:
                print 'Error: save_NNF: ', msg

        return success

    #
    # Print the algorithm parameters
    #
    def print_parameters(self):
        print '-----------------------------------------------------------------'
        print 'PatchMatch parameters:'
        if self._init_NNF_filename is not None:
            nnf_str = self._init_NNF_filename
        else:
            nnf_str = 'Generated internally'
        print '\tInitial NNF: \t\t', nnf_str
        print '\tIterations: \t\t', self.iterations()
        print '\tPatch size: \t\t', self.patch_size()
        print '\tAlpha: \t\t\t', self.alpha()
        print '\tW: \t\t\t', self.w()
        print '\tPropagation enabled: \t', not self.propagation_enabled()
        print '\tRandom search enabled: \t', not self.random_enabled()
        print 'Output path and base filename: \t', self.output()
        output_str = ''
        if self.NNF_vectors():
            output_str += "correspondences, "
        if self.NNF_image():
            output_str += "color nnf, "
        if self.rec_source():
            output_str += "rec'd source "
        print 'Visualization parameters:'
        if len(output_str)>0:
            print '\tOutput files: \t\t', output_str
        print '\tNNF subsampling: \t', self.NNF_subsampling()
        print '\tNNF line width: \t', self.NNF_line_width()
        print '\tNNF line color: \t', self.NNF_line_color()
        print '\tMatplotlib server mode:', self.server()
        print '\tTmp directory: \t\t', self.tmpdir()
        print '-----------------------------------------------------------------'

    #
    # Create a filename that records the algorithm parameters
    #
    def make_filename(self, label, suffix, lastIter=False):
        if not lastIter:
            iter_str = 'iter%s'%(self._current_iteration-1)
        else:
            iter_str = 'last'
        return self.output()+'.%s.p%s.a%s.w%s.prop%s.rand%s.%s.%s'\
                                  %(label,
                                    self.patch_size(),
                                    self.alpha(),
                                    self.w(),
                                    not self.propagation_enabled(),
                                    not self.random_enabled(),
                                    iter_str,
                                    suffix)


    #
    # Execute k iterations of the PatchMatch algorithm and
    # save the results
    #

    def run_iterations(self):
        # initialize the algorithm data structures if this is the first run
        if self._need_init:
            self.initialize_algorithm()
        self.print_parameters()

        ok = True
        while ok:
            ok = self.step_algorithm()

        return

    #
    # Helper methods for setting the algorithm's input, output and control parameters
    #

    # accessor methods for private variables
    def set_iterations(self, i):
        if i >= 0:
            self._iters = i

    def iterations(self):
        return self._iters

    def set_patch_size(self, s):
        if s % 2 == 1:
            # patch sizes must be odd
            self._patch_size = s
            self._need_init = True
        else:
            print 'Warning: Patch size must be odd, reset to %d'%self._patch_size

    def patch_size(self):
        return self._patch_size

    def set_alpha(self, a):
        self._alpha = a

    def alpha(self):
        return self._alpha

    def set_w(self, r):
        if r >= 0:
            self._w = r

    def w(self):
        return self._w

    def set_random(self, val):
        self._random_enabled = val

    def random_enabled(self):
        return self._random_enabled

    def set_propagation(self, val):
        self._propagation_enabled = val

    def propagation_enabled(self):
        return self._propagation_enabled

    def set_init_NNF(self, nnf_file=None):
        if self._images['source'] is None:
            print 'Warning: NNF cannot be loaded before loading a source image'
            return
        if nnf_file is not None:
            ok, f, msg = load_NNF(nnf_file, shape=self._images['source'].shape[0:2])
            if not ok:
                print 'Warning: load_NNF: ',msg
                print 'Generating NNF internally instead'
                self._f = init_NNF(self._images['source'])
                self._init_NNF_filename = None
            else:
                self._init_NNF_filename = nnf_file
                self._f = f
        else:
            self._init_NNF_filename = None
            self._f = init_NNF(self._images['source'])

    def set_output(self, filename):
        self._output = filename

    def output(self):
        return self._output

    def NNF(self):
        return self._f

    def set_im_shape(self):
        if self._images['source'] is not None:
            self._im_shape = self._images['source'].shape
            self._need_init = True

    # variables controlling the image display of NNFs

    def set_server(self, val):
        self._server = val

    def server(self):
        return self._server

    def set_partial_results(self, val):
        self._partial_results = val

    def partial_results(self):
        return self._partial_results

    def set_NNF_subsampling(self, val):
        self._NNF_subsampling = val

    def NNF_subsampling(self):
        return self._NNF_subsampling

    def set_NNF_line_width(self, val):
        self._NNF_line_width = val

    def NNF_line_width(self):
        return self._NNF_line_width

    def set_NNF_line_color(self, val):
        self._NNF_line_color = val

    def NNF_line_color(self):
        return self._NNF_line_color

    def set_NNF_image(self, val):
        self._NNF_image = val

    def NNF_image(self):
        return self._NNF_image

    def set_NNF_vectors(self, val):
        self._NNF_vectors = val

    def NNF_vectors(self):
        return self._NNF_vectors

    def set_rec_source(self, val):
        self._rec_source = val

    def rec_source(self):
        return self._rec_source

    def set_tmpdir(self, tmpdir):
        self._tmpdir = tmpdir

    def tmpdir(self):
        return self._tmpdir
