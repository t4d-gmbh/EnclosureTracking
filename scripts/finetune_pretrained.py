"""This script documents the necessary steps to carry out posture tracking
with a pretrained model and fine-tuning

This tries to follow:
https://deeplabcut.github.io/DeepLabCut/docs/maDLC_UserGuide.html

"""
import os
import glob
import warnings
import deeplabcut as dlc

from enctracking.helpers import (
    parts_mapping,
    to_pretrained_multianimal,
    get_config_path,
)
# ###
# 1. Create a project
# ###
# 1.1 We start with creating a project with a pre-trained
#     model
user = 'ml_user'
working_dir = f"/home/{user}"
project_name = "PretrainedTracker"
model = "superanimal_topviewmouse"
path_to_videos = f"/home/{user}/data/samples"

config_path = get_ocnfig_path(working_dir=working_dir,
                              project_name=project_name,
                              user=user)

# ###
# 2. Label some images
# ###
# NOTE: This is best done using the GUI.
#       Simply open a new terminal and run:
#       `ipython -m deeplabcut`.
#       Then load the project (i.e. open the
#       config.yaml and go to the panel 'label data'
#       (or similar)
# 2.2. Checking the labels
dlc.check_labels(config_path, visualizeindividuals=True)

# ###
# 3. Create the training dataset
# ###
# 3.1. Since we are using a pre-trained model that we aim to fine-tune,
#      we need to first explicitly map our markers with the markers used
#      in the pre-trained model.
#      We are using the same markers, so this step maps each name to itself.
dlc.modelzoo.utils.create_conversion_table(config=config_path,
                                           super_animal='superanimal_topviewmouse',
                                           project_to_super_animal=parts_mapping) 
# 3.2. Now we can create the actuals dataset
dlc.create_multianimaltraining_dataset(config_path, net_type='dlcrnet_ms5',
                                       detector_type='fasterrcnn_mobilenet_v3_large_fpn')

# ###
# 4. Train the network (fine tuning)
# ###
torch_params = dict(
    batch_size=4,
)
dlc.train_network(config=config_path, epochs=None, **torch_params)
