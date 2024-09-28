
import numpy as np
import scipy
import pandas as pd

# import tensorflow as tf
# from tensorflow.keras.callbacks import ModelCheckpoint

from glob import glob
from tifffile import imread
from csbdeep.utils import Path, normalize

from stardist import calculate_extents#, gputools_available
from stardist import Rays_GoldenSpiral
#from stardist.matching import matching_dataset
from stardist.models import Config3D, StarDist3D

np.random.seed(42)

# ## Load and preprocess data

# In[3]:


# retrieve file names

folder = '../DNA_mem9_morula_cropped/'

train_files = sorted(glob(folder + '*.tif'))
label_files = sorted(glob(folder + '3d_segments_final/*.tiff'))
assert all(Path(train_file).name[0:9]==Path(label_file).name[0:9] for train_file, label_file in zip(train_files,label_files))
print(len(label_files))

# In[99]:

# load images

X = list(map(imread,train_files))
Y = list(map(imread,label_files))

n_channel = 1 if X[0].ndim == 3 else X[0].shape[-1]


# In[101]:


# normalize input images and fill holes in labels

axis_norm = (0,1,2)   # normalize channels independently
# axis_norm = (0,1,2,3) # normalize channels jointly
# if n_channel > 1:
#     print("Normalizing image channels %s." % ('jointly' if axis_norm is None or 3 in axis_norm else 'independently'))
#     sys.stdout.flush()

X_norm = [normalize(x,1,99.8,axis=axis_norm) for x in X]
Y_fill = [y.astype(int) for y in Y]

# Y_fill = [fill_label_holes(y.astype(int)) for y in tqdm(Y)]
# diffs = [np.sum(y_fill - y) for y,y_fill in zip(Y,Y_fill)]
# np.sum(diffs) # =0 so we do not need to perform fill_label_holes


# In[22]:


# split into training and validation data

assert len(X_norm) > 1, "not enough training data"
rng = np.random.RandomState(48)
ind = rng.permutation(len(X_norm))
n_val = max(1, int(round(0.15 * len(ind))))
ind_train, ind_val = ind[:-n_val], ind[-n_val:]
X_val, Y_val = [X_norm[i] for i in ind_val]  , [Y_fill[i] for i in ind_val]
X_trn, Y_trn = [X_norm[i] for i in ind_train], [Y_fill[i] for i in ind_train] 
print('number of images: %3d' % len(X))
print('- training:       %3d' % len(X_trn))
print('- validation:     %3d' % len(X_val))



# ## Configure network

# In[160]:


# set up Config3D

print(Config3D.__doc__)

extents = calculate_extents(Y_fill)
anisotropy = tuple(np.max(extents) / extents)
print('empirical anisotropy of labeled objects = %s' % str(anisotropy))

# 96 is a good default choice (see 1_data.ipynb)
n_rays = 96

# # Use OpenCL-based computations for data generator during training (requires 'gputools')
# use_gpu = gputools_available()

# if use_gpu:
#     from csbdeep.utils.tf import limit_gpu_memory
#     # adjust as necessary: limit GPU memory to be used by TensorFlow to leave some to OpenCL-based computations
#     limit_gpu_memory(0.8)
#     # alternatively, try this:
#     # limit_gpu_memory(None, allow_growth=True)

# Predict on subsampled grid for increased efficiency and larger field of view
#grid = tuple(1 if a > 1.5 else 2 for a in anisotropy)

# Use rays on a Fibonacci lattice adjusted for measured anisotropy of the training data
rays = Rays_GoldenSpiral(n_rays, anisotropy=anisotropy)

# U-Net configuration
# FOV goes more or less like: kernel_size + pool_size*kernel_size + pool_size^2*kernel_size + ... + pool_size^depth*kernel_size  
# With 4 layers, kernel_size = 4, pool_size = 2, this is at least 124, which should be enough for our biggest region (113 pixels) in plane. 
# I suggest reducing pool_size to 1 and kernel_size to 3 for the through-plane direction, which should still give a FOV of at least 15 voxels.

conf = Config3D (
    rays             = rays,
    grid             = (1,1,1),
    anisotropy       = anisotropy,
    use_gpu          = False,
    n_channel_in     = n_channel,
    # adjust for your data below (make patch size as large as possible)
    train_patch_size = (40,624,624),
    train_batch_size = 16,
    backbone = 'unet',
    unet_n_depth = 4,
    unet_pool = (1,2,2),
    unet_kernel_size = (3,4,4),
)

# # ResNet configuration
# FOV goes more or less: 4 + floor(kernel_size/2) + n_blocks * n_conv_per_block * floor(kernel_size/2)
# With kernel_size = (2,10,10), n_blocks = 6, n_conv_per_block = 4, the FOV should be = (29,129,129) 
# conf = Config3D (
#     rays             = rays,
#     grid             = (1,1,1),
#     anisotropy       = anisotropy,
#     use_gpu          = use_gpu,
#     n_channel_in     = n_channel,
#     # adjust for your data below (make patch size as large as possible)
#     train_patch_size = (40,624,624),
#     train_batch_size = 16,
#     backbone = 'resnet',
#     resnet_n_blocks = 6,
#     resnet_n_conv_per_block = 4,
#     resnet_kernel_size = (2,10,10)
# )

#print(conf)
#vars(conf)


# In[161]:


# load model

model = StarDist3D(conf, name='stardist', basedir='models')


# In[94]:


# data augmenter

def random_fliprot(img_in, mask_in, axis=None):
    if axis is None:
        axis = tuple(range(mask_in.ndim))
    axis = tuple(axis)

    img = np.copy(img_in)
    mask = np.copy(mask_in)

    assert img.ndim>=mask.ndim
    perm = tuple(np.random.permutation(axis))
    transpose_axis = np.arange(mask.ndim)
    for a, p in zip(axis, perm):
        transpose_axis[a] = p
    transpose_axis = tuple(transpose_axis)
    img = img.transpose(transpose_axis + tuple(range(mask.ndim, img.ndim)))
    mask = mask.transpose(transpose_axis)
    for ax in axis:
        if np.random.rand() > 0.5:
            img = np.flip(img, axis=ax)
            mask = np.flip(mask, axis=ax)

    # Add random angle in +- 45 degree rotation about the through slice axis (combined with the flips, this covers all rotations)
    angle = np.random.uniform(low=-45, high=45)
    #print(angle)
    img_out = np.zeros(img.shape)
    mask_out = np.zeros(mask.shape)
    scipy.ndimage.rotate(img, angle, axes=axis, reshape=False, output=img_out, order=3, mode='constant', cval=0.0, prefilter=True)
    scipy.ndimage.rotate(mask, angle, axes=axis, reshape=False, output=mask_out, order=0, mode='constant', cval=0.0, prefilter=True)

    return img_out, mask_out.astype(int)

def random_intensity_change(img):
    img = img*np.random.uniform(0.6,2) + np.random.uniform(-0.2,0.2)
    return img

def augmenter(x, y):
    """Augmentation of a single input/label image pair.
    x is an input image
    y is the corresponding ground-truth label image
    """
    # Note that we only use fliprots along axis=(1,2), i.e. the yx axis
    # as 3D microscopy acquisitions are usually not axially symmetric
    x, y = random_fliprot(x, y, axis=(1,2))
    x = random_intensity_change(x)
    return x, y


# ## Training and threshold optimisation

# In[139]:


# training

history = model.train(X_trn, Y_trn, validation_data=(X_val,Y_val), augmenter=augmenter,
                      epochs=150, steps_per_epoch=32)


pd.DataFrame(history.history).to_csv('models/stardist/history.csv')


# In[140]:


# threshold optimisation (threshold is for converting the object probability map into labels)

model.optimize_thresholds(X_val, Y_val)









