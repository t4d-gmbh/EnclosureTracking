"""Several post-training steps to evaluste and explore a trained model

This is script 3 in the workflow
"""
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

dlc.evaluate_network(config=config_path, plotting=True)
