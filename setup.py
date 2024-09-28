# -*- coding: utf-8 -*-
"""
Created on Mon Sep 23 07:45:06 2024

@author: Anita Karsa, University of Cambridge, UK
"""

from setuptools import setup, find_packages

setup(
    name='star3d',  # The name of your package
    version='0.0.1',  # Version of your package
    packages=find_packages(),  # Automatically find and include your packages
    # package_data={"optimal3dtracks": ["*.xml"]},
    install_requires=['numpy', 'scipy', 'matplotlib', 'stardist',
                      'scikit-image', 'pandas', 'csbdeep', 'tifffile',
                      'pathlib', 'ipywidgets'],  # External dependencies can be listed here
    description='STAR-3D (STardist-based network for AnisotRopic 3D images) is a trained StarDist-3D-based \
        network for 3D nucleus segmentation in highly anisotropic images.',
    author='Anita Karsa',
    author_email='ak2557@cam.ac.uk',
    url='https://github.com/akarsa/star-3d/',  # The URL of your GitHub repo
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',  # Minimum Python version requirement
)

