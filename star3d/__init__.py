from __future__ import absolute_import, print_function

from stardist.models import StarDist3D

from csbdeep.models import register_model, clear_models_and_aliases
# register pre-trained model
clear_models_and_aliases(StarDist3D)
register_model(StarDist3D, 'STAR-3D', 
               'https://github.com/akarsa/star-3d/releases/download/STAR-3D-2024-08-27/star-3d-weights.zip', 
               '29944b33432496d3616055a703ddd08918f710ba12aa8924eb14682d1b0eef78')

del register_model, clear_models_and_aliases
