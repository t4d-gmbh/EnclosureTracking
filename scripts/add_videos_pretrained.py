"""This script adds new video files to an existing project

This is (an optinal) 2nd step after runnig hte init_pretrained.py script

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
user = 'ml_user'
working_dir = f"/home/{user}"
project_name = "PretrainedTracker"
path_to_videos = f"/home/{user}/data/samples"

config_path = get_config_path(working_dir=working_dir,
                              project_name=project_name,
                              user=user)

videos_to_add = [ 
    os.path.join(path_to_videos, 'rec_148__2023-06-21__00-08-00_2.mp4'),
    os.path.join(path_to_videos, 'rec_148__2023-06-21__00-08-00_3.mp4'),
                 
    os.path.join(path_to_videos, 'rec_148__2023-06-21__07-28-00_2.mp4'),
    os.path.join(path_to_videos, 'rec_148__2023-06-21__07-28-00_3.mp4'),
    os.path.join(path_to_videos, 'rec_148__2023-06-21__07-28-00_4.mp4'),
]

dlc.add_new_videso(config=config_path,
                   videos=videos_to_add)
