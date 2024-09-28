"""
Created on Tue Mar 13 2024

@author: Anita Karsa, University of Cambridge, UK
"""

# In[]: Import all necessary tools

import os
import numpy as np
import scipy

from glob import glob
from tifffile import imread,imwrite

from stardist.models import StarDist3D

from skimage import measure

from star3d.utils_3Dsegmentation import (  # noqa: E402
    patch_segmentation,
    norm_X
)

# In[]: Data directory and results directories

folder = 'Embryo1/'
intensity_folder = 'intensity'
segmentation_results = 'STAR-3D_results'
filtered_results = 'STAR-3D_filtered'

if not os.path.isdir(folder + segmentation_results):
    os.makedirs(folder + segmentation_results)
if not os.path.isdir(folder + filtered_results):
    os.makedirs(folder + filtered_results)
    
# In[]: Set parameters
    
resolution = np.array([2,0.174,0.174]) # in um
# We have found the network to perform optimally when the objects to be detected are about 5 voxels in the lateral direction
# and 60 voxels axially
expected_object_diameter = 10 # um
optimal_resolution = expected_object_diameter/np.array([5.0,60.0,60.0])
downsampling = resolution/optimal_resolution
gamma_factor = 1.0

# In[]: Load intensity files and model

# Load intensity image files
input_files = sorted(glob(folder + intensity_folder + '/*.tif'))[:]

# Load model
model = StarDist3D.from_pretrained('STAR-3D')


# In[]: Apply STAR-3D to the images (this might take a very long time)
    
for file in input_files:
    
    print(file)
    
    # Name the output file (this is normally part of the intensity image file name)
    output_file_name = os.path.splitext(os.path.basename(file))[0][:] + '_label.tif'

    # Load image    
    intensity = imread(file,is_ome=False)
    
    # Crop intensity image if possible. Applying STAR-3D can take a very long time so it is better to avoid 
    # segmenting empty parts of the image wherever possible. 
    # We have an embryo segmentation method currently under development which could automate this step
    crop = np.array([[0,intensity.shape[0]],[0,intensity.shape[1]],[0,intensity.shape[2]]])
    intensity_cropped = intensity[crop[0,0]:crop[0,1],crop[1,0]:crop[1,1],crop[2,0]:crop[2,1]]
    
    # Normalise, downsample, and gamma correct image
    X_norm = norm_X(intensity_cropped,downsampling,gamma_factor)

    # Apply STAR-3D in patches to avoid running into Tensorflow's tensor size limit
    # Here we calculated the number of patches from the array size, but this method is not very reliable
    # Increase patches if tensorflow returns an error
    patches = 2**(X_norm.size>np.array([6e7,5e7,4e7])).astype(int) + 2**(X_norm.size>np.array([29e7,24e7,19e7])).astype(int) - 1
    print(patches)
    # Set margins where patches overlap to be bigger than the nuclei
    margins = 40/(resolution/downsampling) # we expect the nuclei to be no more than 40 um in diameter
    
    # Apply STAR-3D and resample label map to the original resolution
    Y_pred = scipy.ndimage.zoom(patch_segmentation(X_norm,model,patches,margins),
                                np.asarray(intensity_cropped.shape)/np.asarray(X_norm.shape),order = 0)
    
    # Separate different regions with the same label 
    # Although it shouldn't happen that two distinct regions have the same label, it's better to make sure.
    Y_pred = measure.label(Y_pred.astype(int), connectivity=2)
    
    # Pad Y_pred to original size (especially if you're planning to use our tracking methods as well)
    Y_pred_full = np.zeros(intensity.shape)
    Y_pred_full[crop[0,0]:crop[0,1],crop[1,0]:crop[1,1],crop[2,0]:crop[2,1]] = Y_pred
    
    # Save unfiltered result
    imwrite(folder + segmentation_results + '/' + output_file_name, Y_pred_full.astype('uint16'), imagej=True, 
           resolution=1/resolution[1:3],
           metadata={
               'spacing': resolution[0],
               'unit': 'um',
               'axes': 'ZYX'
           })

    