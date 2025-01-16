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
dlc.create_pretrained_project(
    project=project_name,
    experimenter=user,
    videos=[path_to_videos],
    working_directory=working_dir,
    model=model,
    copy_videos=False,
    filtered=False,  # False: frame-by-frame predictions
    createlabeledvideo=True,
    trainFraction=0.95  # 0.95 is the default
)

config_path = get_config_path(working_dir=working_dir,
                              project_name=project_name,
                              user=user)


to_pretrained_multianimal(config_file=config_path, nbr_animals=12)

# Now we can go ahead and label data
