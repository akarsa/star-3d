"""
Created on Wed March 13 2024

@author: Anita Karsa, University of Cambridge, UK
"""

# In[]: Import all necessary tools

import numpy as np
import scipy
import matplotlib.pyplot as plt

from skimage import measure
import pandas as pd

from csbdeep.utils import normalize

# In[]

################### Helper functions for display and outputs ###################


# In[]: Display 3D + time image with segmentation contours

import ipywidgets as widgets
from ipywidgets import interact


def show_4d_with_contours(im,seg):
    """
    Visualize a 4D grayscale image with contour overlays of segmented regions.

    Args:
        im (ndarray): 4D grayscale image normalized between 0 and 1.
        seg (ndarray): 4D integer label map.

    Returns:
        None
    """
    
    # Extract dimensions of the input image
    n_timepoints = im.shape[0]
    n_slices = im.shape[1]

    # Erode the segmentation mask to create contours
    edges = scipy.ndimage.binary_erosion(seg, np.ones((1, 1, 5, 5), np.uint8), iterations=1)
    edges = seg * (1 - edges)
    max_label = np.max(edges)
    
    # Define an update function for interactive visualization
    def update(timepoint, slice_num):
        timepoint = np.round(timepoint)
        slice_num = np.round(slice_num)
        
        # Plot the original image
        plt.figure()
        plt.imshow(im[int(timepoint), int(slice_num), :, :], cmap='gray', interpolation='none', vmin=0, vmax=255)
        
        # Plot the contours overlaid on the original image
        plt.imshow(edges[int(timepoint), int(slice_num), :, :], vmax=max_label, cmap='prism', \
                   alpha=0.5*(edges[int(timepoint), int(slice_num), :, :] > 0), interpolation='none')
        
        # Add label numbers to the plot
        df = pd.DataFrame(measure.regionprops_table(edges[int(timepoint), int(slice_num), :, :].astype(int), \
                                                     properties=["label", "centroid"]))
        for i in range(len(df)):
            plt.text(df["centroid-1"][i], df["centroid-0"][i], str(df["label"][i]), color='white', fontsize=12)
        
        # Show the plot
        plt.show()
    
    # Create sliders for timepoint and slice selection
    timepoint_widget = widgets.FloatSlider(
        value=int(n_timepoints/2),
        min=0,
        max=n_timepoints-1,
        step=1,
        description='Time: ',
        disabled=False,
        continuous_update=True,
        orientation='horizontal',
        readout=True,
        readout_format='',
        style={'description_width': 'initial'}
    )
    slice_widget = widgets.FloatSlider(
        value=int(n_slices/2),
        min=0,
        max=n_slices-1,
        step=1,
        description='Slice: ',
        disabled=False,
        continuous_update=True,
        orientation='horizontal',
        readout=True,
        readout_format='',
        style={'description_width': 'initial'}
    )
    
    # Create an interactive widget to update the visualization
    interact(update, timepoint=timepoint_widget, slice_num=slice_widget)
    

# In[]: Colour segmentations according to their position in the morula (from edge (blue) to middle (red))

def distance_from_edge_colouring(label_image, label_file, anisotropy):
    """
    Computes a distance map from the edges of labeled regions in a 3D image volume.

    Args:
        label_image (ndarray): A 3D array representing labeled regions
        label_file (str): Label file name (this is just printed at the beginning)
        anisotropy (int): Anisotropy factor for erosion

    Returns:
        list: A list containing two items: the distance map and the distance map masked by the label image. 
    """
    
    print(label_file)  # Print the label file name
        
    # Create distance map
    mask = convex_hull_3D(label_image).astype(int)
    distance_map = np.zeros(label_image.shape)
    dist = 1
    
    # Erode the mask iteratively
    while np.sum(mask) > 0:
        for i in range(anisotropy - 1):
            eroded = scipy.ndimage.binary_erosion(mask, np.ones((1, 3, 3), np.uint8), iterations=1).astype(int)
            distance_map += (mask - eroded) * dist
            mask = eroded.copy()
            dist += 1
        eroded = scipy.ndimage.binary_erosion(mask, np.ones((3, 3, 3), np.uint8), iterations=1)
        distance_map += (mask - eroded) * dist
        mask = eroded.copy()
        dist += 1
    
    # Rescale distance map between 2 (inside) and 1 (outside)
    distance_map = (1 - distance_map) / (np.max(distance_map) - 1) + 2
    
    # Mask the distance map by the label image
    distance_map_masked = distance_map * label_image.astype(bool)
    
    return distance_map, distance_map_masked

def convex_hull_3D(label_image):
    """
    Compute the convex hull of labeled regions in a 3D image volume.

    Args:
        label_image (ndarray): A 3D array representing labeled regions.

    Returns:
        ndarray: A binary mask indicating the convex hull of the labeled regions.
    """

    # Get coordinates of all non-zero points in the label image
    coordinates = np.transpose(np.concatenate(np.expand_dims(np.where(label_image > 0), 1), axis=0))
    
    # Compute the convex hull of the non-zero points
    hull = scipy.spatial.ConvexHull(coordinates)
    deln = scipy.spatial.Delaunay(coordinates[hull.vertices])
    
    # Generate indices for 2D and 3D arrays
    idx_2d = np.indices(label_image.shape[1:], np.int16)
    idx_2d = np.moveaxis(idx_2d, 0, -1)
    idx_3d = np.zeros((*label_image.shape[1:], label_image.ndim), np.int16)
    idx_3d[:, :, 1:] = idx_2d
    
    # Initialize a mask for each slice
    mask = np.zeros_like(label_image, dtype=bool)
    
    # Iterate over slices to find points inside the convex hull
    for z in range(len(label_image)):
        idx_3d[:,:,0] = z
        s = deln.find_simplex(idx_3d)
        mask[z, (s != -1)] = 1
        
    return mask

def project_inside_edge(label_image):
    """
    Project the regions of a labeled 3D image onto a 2D projection colour-coded 
    as inside (red) and edge (blue) regions.

    Args:
        label_image (ndarray): A 3D array representing labeled regions.

    Returns:
        ndarray: A colour-coded 2D projection of the regions.
    """
    
    # Create a projection array with three channels (RGB)
    projection = np.zeros([label_image.shape[1], label_image.shape[2], 3])
    projection_red = projection[:, :, 0]  # Red channel
    projection_blue = projection[:, :, 2]  # Blue channel
    
    # Compute the sum of label values along each slice
    slices = np.sum(label_image, axis=(1, 2))
    slices = np.where(slices > 0)
    
    # Determine the starting and ending slices for projection
    slice_0 = np.max([slices[0][0] - 10, 0])
    slice_end = np.min([slices[0][-1] + 10, label_image.shape[0] - 1])
    
    # Compute the total number of slices
    num_slices = slice_end - slice_0 + 1
    
    # Project regions inside the edge onto the projection
    for s in range(int(slice_0), int(slice_end + 1)):
        current_slice = label_image[s, :, :].copy()
        # Compute the red channel values
        projection_red[current_slice > 0] = (2 - current_slice[current_slice > 0]) * (s + 1) / num_slices
        # Compute the blue channel values
        projection_blue[current_slice > 0] = (current_slice[current_slice > 0] - 1) * (s + 1) / num_slices
    
    return projection
    
# In[]: Top-down view plot of the segmented cells

def project_colours(label_image):
    """
    Project the colors of labeled regions in a 3D image volume onto a 2D projection.

    Args:
        label_image (ndarray): A 3D array representing labeled regions.

    Returns:
        ndarray: A 2D projection of the colors of labeled regions.
    """
    
    # Create a projection array with three channels (RGB)
    projection = np.zeros([label_image.shape[1], label_image.shape[2], 3])
    projection_red = projection[:, :, 0]  # Red channel
    projection_green = projection[:, :, 1]  # Green channel
    projection_blue = projection[:, :, 2]  # Blue channel
    
    # Compute the sum of label values along each slice
    slices = np.sum(label_image, axis=(1, 2))
    slices = np.where(slices > 0)
    
    # Determine the starting and ending slices for projection
    slice_0 = np.max([slices[0][0] - 10, 0])
    slice_end = np.min([slices[0][-1] + 10, label_image.shape[0] - 1])
    
    # Create a colormap
    cmap = np.zeros([1000, 3])
    for c in range(1000):
        np.random.seed(c)
        cmap[c, :] = np.random.uniform(low=0.0, high=1.0, size=[1, 3])
    
    # Project the colors of labeled regions onto the projection
    for s in range(int(slice_0), int(slice_end + 1)):
        current_slice = label_image[s, :, :].copy()
        # Red channel
        projection_red[current_slice > 0] = cmap[current_slice[current_slice > 0] % 1000, 0]
        # Green channel
        projection_green[current_slice > 0] = cmap[current_slice[current_slice > 0] % 1000, 1]
        # Blue channel
        projection_blue[current_slice > 0] = cmap[current_slice[current_slice > 0] % 1000, 2]

    return projection

# In[]

################### Helper functions for segmentation ###################

# In[]: Normalising, downsampling, and gamma correction of the intensity images
    
def norm_X(X, downsampling, gamma_factor):
    """
    Normalize and apply gamma correction to a 3D array.

    Args:
        X (ndarray): A 3D array representing input data.
        downsampling (tuple): Tuple specifying downsampling factors along each dimension.
        gamma_factor (float): Gamma factor for gamma correction.

    Returns:
        ndarray: Normalized and gamma-corrected 3D array.
    """
    
    # Check if downsampling is all ones
    if all(downsampling == 1):
        # Normalize each array in X and store the results in X_norm
        X_norm = normalize(X, 1, 99.8, axis=(0, 1, 2))
    else:
        # Normalize each array in X, apply downsampling, and store the results in X_norm
        X_norm = scipy.ndimage.zoom(normalize(X, 1, 99.8, axis=(0, 1, 2)), downsampling)
    
    # Apply gamma correction to each array in X_norm
    X_norm = np.power(np.clip(X_norm, 0, None), gamma_factor) 
    
    return X_norm
    
# In[]: Apply model in patches to avoid running into Tensorflow's tensor size limit

def patch_segmentation(X_norm, model, patches, margins):
    """
    Apply model to input data (X_norm) using a patch-based approach.

    Args:
        X_norm (ndarray): Normalized input data (3D).
        model (object): Segmentation model.
        patches (tuple): Number of patches along each dimension.
        margins (tuple): Margins for each patch. 
                         This should be bigger than the largest expected region

    Returns:
        ndarray: Label map.
    """

    # Initialize the output array
    Y_pred = np.zeros(X_norm.shape).astype(int)

    size = np.asarray(Y_pred.shape)

    # Loop through all patches
    max_value = 0
    for i in range(patches[0]):
        for j in range(patches[1]):
            for k in range(patches[2]):

                print([i, j, k])

                # Determine start and end voxels in all directions
                start = np.clip(size/patches * np.array([i, j, k]) - margins/2, a_min=0, a_max=None).astype(int)
                end = np.clip(size/patches * (np.array([i, j, k]) + 1) + margins/2, a_min=None, a_max=size).astype(int)

                # Apply model to the patch
                y_pred = model.predict_instances(X_norm[start[0]:end[0], start[1]:end[1], start[2]:end[2]])[0].astype(int)

                # Remove labels on the edge of the patch (but only if it's an inside edge)
                mask = np.zeros(y_pred.shape)
                mask[np.array([0, -1])[~np.isin([start[0], end[0]], [0, size[0]])], :, :] = 1
                mask[:, np.array([0, -1])[~np.isin([start[1], end[1]], [0, size[1]])], :] = 1
                mask[:, :, np.array([0, -1])[~np.isin([start[2], end[2]], [0, size[2]])]] = 1

                labels = np.unique(y_pred[mask == 1])
                y_pred[np.isin(y_pred, labels)] = 0

                # Increase all labels to avoid having the same label
                y_pred += y_pred.astype(bool).astype(int) * max_value
                
                # Remove labels from the target that we'll add from this patch
                # to avoid each cell having several labels due to slight mismatches
                Y_pred_patch = Y_pred[start[0]:end[0], start[1]:end[1], start[2]:end[2]]
                labels = np.unique(Y_pred_patch[mask == 1])
                Y_pred_patch[~np.isin(Y_pred_patch, labels)] = 0

                # Insert y_pred into Y_pred
                Y_pred_patch += y_pred

                # Calculate new max_value
                max_value = np.max(Y_pred)

    return Y_pred

# In[]

################### Helper functions for filtering ###################

# In[]: Masking the embryo by fitting a sphere around it using 3D Hough transforms 
# This actually works pretty well if the embryo is spherical. Unfortunately, this is not always the case.
    
def Hough_circle_3D(X_norm, hough_radii, hough_resolution):
    """
    Perform 3D Hough transform to detect spherical embryonic boundaries.

    Args:
        X_norm (ndarray): Normalized input data (3D).
        hough_radii (ndarray): Radii of circles to be detected (in um).
        hough_resolution (ndarray): Resolution of X_norm (usually in um).

    Returns:
        ndarray: 3D binary array representing the embryonic boundary.
    """

    Hough_transform = []

    # Calculate size and middle of the Hough transform space
    circle_size = (2.2 * np.max(hough_radii) / hough_resolution)
    middle = (circle_size) / 2

    # Generate meshgrid for Hough transform space
    z, x, y = np.meshgrid(np.arange(int(circle_size[0])), np.arange(int(circle_size[1])), np.arange(int(circle_size[2])), indexing='ij')
    z = (z.astype(float) - middle[0]) * hough_resolution[0]
    x = (x.astype(float) - middle[1]) * hough_resolution[1]
    y = (y.astype(float) - middle[2]) * hough_resolution[2]
    r2 = z**2 + x**2 + y**2

    del x, y, z, middle

    # Perform Hough transform for each radius
    for radius in hough_radii:
        circle = np.zeros(circle_size.astype(int))
        circle[r2 < radius**2] = 1
        Hough_transform.append(scipy.signal.fftconvolve(X_norm, circle, mode='same') / np.sum(circle))

    Hough_transform = np.stack(Hough_transform, axis=3)

    # Find optimal radius
    max_per_radii = np.diff(np.max(Hough_transform, axis=(0, 1, 2)), n=2)
    ind = np.asarray(np.where(max_per_radii == np.min(max_per_radii))) + 1

    # Generate embryonic boundary
    coords = np.where(Hough_transform[:, :, :, ind] == np.max(Hough_transform[:, :, :, ind]))
    z, x, y = np.meshgrid(np.arange(X_norm.shape[0]), np.arange(X_norm.shape[1]), np.arange(X_norm.shape[2]), indexing='ij')
    z = (z.astype(float) - np.mean(coords[0])) * hough_resolution[0]
    x = (x.astype(float) - np.mean(coords[1])) * hough_resolution[1]
    y = (y.astype(float) - np.mean(coords[2])) * hough_resolution[2]
    r2 = z**2 + x**2 + y**2
    embryo_boundary = np.zeros(X_norm.shape)
    embryo_boundary[r2 < hough_radii[ind]**2] = 1

    return embryo_boundary

# In[]: Supervised filtering of regions in a specific area of the image

def show_dust(im, seg, label, x):
    """
    Display potential dust particles in an image along with the corresponding segmentation.

    Args:
        im (ndarray): The input image.
        seg (ndarray): The segmentation of the image.
        label (int): The label of the dust particles to be shown.
        x (int): The index of the slice to be displayed.

    Returns:
        None
    """
    # Create a binary segmentation mask for the specified label
    seg_lab = np.zeros(seg.shape)
    seg_lab[seg == label] = 1

    # Generate edges of the segmented dust particles
    edges = scipy.ndimage.binary_erosion(seg_lab, np.ones((1, 5, 5), np.uint8), iterations=1)
    edges = seg_lab * (1 - edges)

    # Determine the maximum label value in the segmentation
    max_label = np.max(seg)
    
    # Create a figure with two subplots
    fig, axs = plt.subplots(1, 2)
    fig.set_size_inches(15, 25)

    # Plot the original image with overlaid edges of the dust particles
    axs[0].imshow(im[int(x), :, :], cmap='gray', interpolation='none')
    axs[0].imshow(edges[int(x), :, :], vmax=1, cmap='prism', alpha=0.5 * (edges[int(x), :, :] > 0), interpolation='none')

    # Plot the segmentation of the dust particles
    axs[1].imshow(seg[int(x), :, :], cmap='prism', vmax=max_label, alpha=0.5 * (seg[int(x), :, :] > 0), interpolation='none')

    # Show the plot
    plt.show()

def filter_regions_around_specific_voxel(label_image, intensity_image, resolution, voxel, radius):
    """
    Filter regions around a specific voxel based user input.

    Args:
        label_image (ndarray): A 3D array representing labeled regions.
        intensity_image (ndarray): A 3D array representing intensity values.
        voxel (ndarray): Coordinates of the voxel of interest.
        resolution (ndarray): Resolution of intensity_image (usually in um).
        radius (float): The radius within which regions are considered (usually in um).

    Returns:
        ndarray: A copy of the label_image with filtered regions.
    """
    # Create a copy of the label image for filtering
    filtered = np.copy(label_image)
    
    # Compute properties of labeled regions
    props = pd.DataFrame(measure.regionprops_table(filtered, properties=["label", "centroid", "area"]))
    
    # Calculate distance of each region's centroid to the specified voxel
    props['distance'] = np.linalg.norm((props[['centroid-0', 'centroid-1', 'centroid-2']].to_numpy() - np.reshape(voxel, [1, 3])) * resolution, axis=1)
    
    # Iterate through labeled regions within the specified radius
    for label in props['label'][(props['distance'] <= radius)].to_numpy():
        dust = ''
        
        # Prompt user to identify if the region is a dust particle or dirt
        while dust not in ['n', 'y']:
            show_dust(intensity_image, label_image, label, props['centroid-0'][props['label'] == label].to_numpy())
            dust = input('Is this a dust particle or dirt? size = ' + str(props['area'][props['label'] == label].to_numpy()) + ' y/n')
        
        # Remove the region from the filtered image if identified as a dust particle
        if dust == 'y':
            filtered[filtered == label] = 0
    
    return filtered

# In[]: Supervised filtering of regions touching across a large surface

def show_connection(im, seg, labels, x):
    """
    Visualize connections between segmented regions and their labels.

    Args:
        im (ndarray): Input image.
        seg (ndarray): Label image.
        labels (list): List of labels to visualize.
        x (int): Slice index to visualize.

    Returns:
        None
    """
    
    # Create a copy of the label image and mask out labels not in the given list
    seg_lab = np.copy(seg)
    seg_lab[~np.isin(seg_lab, labels)] = 0
    
    # Generate edges of the segmented regions
    edges = scipy.ndimage.binary_erosion(seg_lab, np.ones((1, 5, 5), np.uint8), iterations=1)
    edges = seg_lab * (1 - edges)
    
    # Compute the maximum label value
    max_label = np.max(seg)
    
    # Create a figure with two subplots
    fig, axs = plt.subplots(1, 2)
    fig.set_size_inches(15, 25)
    
    # Plot the original image with edges
    axs[0].imshow(im[int(x), :, :], cmap='gray', interpolation='none')
    axs[0].imshow(edges[int(x), :, :], vmax=max_label, cmap='prism', alpha=0.5 * (edges[int(x), :, :] > 0), interpolation='none')
    
    # Plot the segmented image with transparency
    axs[1].imshow(seg[int(x), :, :], cmap='prism', vmax=max_label, alpha=0.5 * (seg[int(x), :, :] > 0), interpolation='none')
    
    # Show the plot
    plt.show()
    

def sort_out_interfacing_regions(label_image, intensity_image, minimum_percentage_of_connection, label_file):
    """
    Identify and manually sort out interfacing regions in the label image.

    Args:
        label_image (ndarray): 3D array representing labeled regions.
        intensity_image (ndarray): 3D array representing intensity values.
        minimum_percentage_of_connection (float): the minimum percentage of surface area where 
                    two regions need to be connected to perform manual checks. Below this, all
                    connections are assumed to be between separate regions.
        label_file: Filename of label image (this is just printed at the beginning).

    Returns:
        ndarray: Modified labeled image with sorted interfacing regions.
    """
    
    # Create a copy of the label image
    filtered = np.copy(label_image)
    
    # Extract unique labels excluding background label (0)
    labels = np.unique(filtered)
    labels = labels[labels != 0]

    # Find combined and separated edges of the segmentations
    combined_edges = filtered * (1 - scipy.ndimage.binary_erosion(filtered, np.ones((1, 3, 3), np.uint8), iterations=1))
    separated_edges = np.zeros(combined_edges.shape)
    for label in labels:
        single_region = np.zeros(filtered.shape)
        single_region[filtered == label] = label
        separated_edges += single_region * (1 - scipy.ndimage.binary_erosion(single_region, np.ones((1, 3, 3), np.uint8), iterations=1))

    # The difference contains interfacing borders between regions
    diff_edges = separated_edges - combined_edges

    # Find interfacing pairs of regions
    labels = np.unique(diff_edges)
    labels = labels[labels != 0]
    interfacing_pairs = []
    centre_slice = []
    connection_edges = np.copy(diff_edges)
    for lab in labels:
        # Dilate the connecting edge of region of label lab
        mask = np.zeros(connection_edges.shape)
        mask[connection_edges == lab] = 1
        mask = scipy.ndimage.binary_dilation(mask, np.ones((1, 3, 3), np.uint8), iterations=1)
        # Find neighboring labels using the dilated mask
        borders = pd.DataFrame(measure.regionprops_table((connection_edges * mask).astype(int), properties=["label", "centroid"]))
        neighbours = borders['label'].to_numpy()
        neighbours = neighbours[~np.isin(neighbours, lab)]
        # Add all neighbours to interfacing pairs
        for neighbour in neighbours:
            interfacing_pairs.append([lab.astype(int), neighbour.astype(int)])
            # Add centre slice
            centre_slice.append(borders['centroid-0'][borders['label'] == neighbour].to_numpy().astype(int))

        # Delete lab from connection_edges (so that all pairs appear only once in the list)
        connection_edges[connection_edges == lab] = 0

    # Examine all borders separately
    for pair, c_slice in zip(interfacing_pairs, centre_slice):

        # Compute the percentage of connection
        percentage_of_connection = np.max([np.sum(diff_edges == pair[0]) / np.sum(separated_edges == pair[0]),
                                           np.sum(diff_edges == pair[1]) / np.sum(separated_edges == pair[1])])

        print('Surface overlap = ' + str(percentage_of_connection * 100) + '%')
        print('Original labels = ' + str(pair))
        if percentage_of_connection > minimum_percentage_of_connection:
            same_region = ''

            while same_region not in ['n', 'y']:
                print('Slice number = ', str(c_slice))
                show_connection(intensity_image, filtered, pair, c_slice)

                # Ask user if the regions are the same
                same_region = input('Are these the same region? y/n')

            if same_region == 'y':
                lab_float = np.median(filtered[diff_edges == pair[1]])
                lab_target = np.median(filtered[diff_edges == pair[0]])
                filtered[filtered == lab_float] = lab_target
                
    return filtered

