<p align="center"><img src="docs/images/logo.png" width="500" /></p>

*Copyright (c) 2023-2024 Anita Karsa, University of Cambridge, UK*

*STAR-3D is distributed under the terms of the GNU General Public License*

ABOUT
-------------------------------------------------------------------------------
STAR-3D (STardist-based network for AnisotRopic 3D images) is a trained StarDist-3D-based 
network for 3D nucleus segmentation in highly anisotropic images. The network was trained on 
annotated mouse embryo images (close to 100 time points) acquired using light-sheet microscopy 
with an in-plane resolution of 0.174 um and a through-plane resolution of 2 um. 

The network (U-Net) was configured with 4 layers, convolutional kernels of size (3,4,4) and 
pooling kernels of size (1,2,2) to ensure a large enough, anisotropic receptive field. On top 
of the built-in augmentation methods of flipping and intensity change, random rotations 
between ±45 degrees about the through-slice axis (i.e. in plane) was added.
See star-3d/training/StarDist_3D_training.py for more details on the network configuration and training.

Our companion method for optimal cell tracking is downloadable from: https://github.com/akarsa/optimal3dtracks

![](docs/images/examples.gif)

INSTALLATION AND SYSTEM REQUIREMENTS
-------------------------------------------------------------------------------
First, install required packages (see DEPENDENCIES or Requirements.txt). 

Use `pip install star3d` to install STAR-3D and download the network weights. 
This should take up to a few minutes.

The software has been extensively tested on a standard computer with Windows 11 OS and
an Intel(R) Core(TM) i9-10980XE CPU @ 3.00GHz processor with 128 GB RAM, but it is
likely to work on different systems and with less memory too.
 
The version of each dependency used for testing can be found in DEPENDENCIES and 
Requirements.txt, but the software is likely to work with other versions as well.

DEMO
-------------------------------------------------------------------------------
demo/demo.ipynb is a short demo performing segmentation in the five time points in 
demo/intensity. Depending on your computer, this might take a while (5-20 minutes).
If memory issues occur, try increasing the number of patches in one or more dimensions
(e.g. `patches = np.array([1,2,2])`). The demo code should produce the below image 
while saving all results in demo/STAR-3D_results. 

<p align="center"><img src="demo/segmentation.png" width="500" /></p>

HOW TO USE
-------------------------------------------------------------------------------

**To perform 3D nucleus segmentation** (see star3d/example.ipynb and star3d/example.py):

```
# 1. Calculate optimal resolution and downsampling factors 
resolution = np.array([2,0.174,0.174]) # in um
# We have found the network to perform optimally when the objects to be detected are about 5 voxels in the lateral direction
# and 60 voxels axially
expected_object_diameter = 10 # um
optimal_resolution = expected_object_diameter/np.array([5.0,60.0,60.0])
downsampling = resolution/optimal_resolution

# 2. Load network
model = StarDist3D.from_pretrained('STAR-3D')

# 3. Load and preprocess intensity image    
intensity = imread(file,is_ome=False)
X_norm = norm_X(intensity,downsampling,1.0)

# 4. Perform segmentation in patches due to large RAM requirements of the network
patches = np.array([1,2,2]) # increase if tensorflow returns out-of-memory error
margins = 40/(resolution/downsampling) # we expect the nuclei to be no more than 40 um in diameter
Y_pred = patch_segmentation(X_norm,model,patches,margins)
Y_pred = scipy.ndimage.zoom(Y_pred,np.asarray(intensity.shape)/np.asarray(X_norm.shape),order = 0)
```

**For additional filtering and display features** (see star3d/example.ipynb):


HOW TO ACKNOWLEDGE
-------------------------------------------------------------------------------
@software{star-3d,

  author       = {Anita Karsa},

  title        = {{STAR-3D}},

  month        = feb,

  year         = 2024,

  url 	       = {https://github.com/akarsa/star-3d}

}

@article{karsa2024star3d,

  title={STAR-3D and Optimal3dTracks: Advanced Deep Learning and Optimal Transport Techniques for Automated 3D Segmentation and Tracking in Pre-Implantation Embryos (manuscript under preparation)},

  author={Karsa, Anita and Boulanger, Jerome and Abdelbaki, Ahmed and Niakan, Kathy K. and Muresan, Leila},

}

DEPENDENCIES
-------------------------------------------------------------------------------
Python (3.11.4)

numpy (1.24.3) (https://numpy.org)

scipy (1.10.1) (https://scipy.org)

matplotlib (3.7.1) (https://matplotlib.org)

scikit-image (0.20.0) (https://scikit-image.org)

pandas (1.5.3) (https://pandas.pydata.org)

stardist (0.8.5) (https://github.com/stardist/stardist)

csbdeep (0.7.4) (https://github.com/CSBDeep/CSBDeep)

tifffile (2023.2.28) (https://github.com/cgohlke/tifffile)

pathlib (1.0.1) (https://github.com/budlight/pathlib)

ipywidgets (8.0.4) (https://github.com/jupyter-widgets/ipywidgets)


CONTACT INFORMATION
-------------------------------------------------------------------------------
Anita Karsa, Ph.D.

Cambridge Advanced Imaging Centre

Dept. of Physiology, Development, and Neuroscience

University of Cambridge,

Cambridge, UK

ak2557@cam.ac.uk
