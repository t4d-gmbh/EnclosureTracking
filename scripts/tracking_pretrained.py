"""This script performs tracking of individuals in a video

This is script 4 in the workflow

..note::
    Running this only realy makes sense if the pose estimation results are not to bad
    See/run [evaluate_pretrained.py](./evaluate_pretrained.py) for details.
"""
import os
import deeplabcut as dlc

from enctracking.helpers import (
    parts_mapping,
    to_pretrained_multianimal,
    get_config_path,
)

user = 'ml_user'
working_dir = f"/home/{user}"
project_name = "PretrainedTracker"
model = "superanimal_topviewmouse"
path_to_videos = f"/home/{user}/data/samples"

config_path = get_config_path(working_dir=working_dir,
                              project_name=project_name,
                              user=user)

videos_to_analyze = [
    os.path.join(path_to_videos, 'rec_148__2023-06-21__00-08-00.mp4'),
]

# Run the analysis of the videos
scorername = dlc.analyze_videos(config=config_path,
                                videos=videos_to_analyze,
                                batch_size=4)
# Create some annotated videos to check the perfomrance
dlc.create_video_with_all_detections(config=config_path,
                                     videos=videos_to_analyze)
