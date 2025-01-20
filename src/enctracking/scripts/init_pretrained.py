"""This script documents the necessary steps to carry out posture tracking
with a pretrained model and fine-tuning.

This tries to follow:
https://deeplabcut.github.io/DeepLabCut/docs/maDLC_UserGuide.html
"""

import os
import glob
import warnings
import deeplabcut as dlc
import argparse

from ..helpers import (
    parts_mapping,
    to_pretrained_multianimal,
    get_config_path,
)

def main(user, working_dir, project_name, model, path_to_videos):
    """Create a DeepLabCut project using a pretrained model and prepare it for labeling.

    This function performs the following steps:
    1. Creates a new DeepLabCut project with the specified parameters.
    2. Configures the project to use a pretrained model.
    3. Prepares the project for labeling by setting up the necessary configurations.

    Args:
        user (str): The username of the experimenter for the project.
        working_dir (str): The directory where the project will be created.
        project_name (str): The name of the project to be created.
        model (str): The name of the pretrained model to be used.
        path_to_videos (str): The path to the video files that will be used for training.

    Returns:
        None: This function does not return any value. It performs actions to create
        and configure the DeepLabCut project.
    
    Raises:
        Exception: If there is an error during project creation or configuration.
    """
    # Create a project with a pre-trained model
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Posture tracking with a pretrained model.")
    parser.add_argument('--user', type=str, default='ml_user', help='Username for the project.')
    parser.add_argument('--working_dir', type=str, default='/home/ml_user', help='Working directory for the project.')
    parser.add_argument('--project_name', type=str, default='PretrainedTracker', help='Name of the project.')
    parser.add_argument('--model', type=str, default='superanimal_topviewmouse', help='Pretrained model to use.')
    parser.add_argument('--path_to_videos', type=str, default='/home/ml_user/data/samples', help='Path to the videos.')

    # Add example usage to the help message
    parser.epilog = (
        "Example usage:\n"
        "  python your_script.py --user new_user --working_dir /home/new_user --project_name NewTracker\n"
        "  python your_script.py --model another_model --path_to_videos /path/to/videos\n"
        "  python your_script.py  # Use default values"
    )

    args = parser.parse_args()

    main(args.user, args.working_dir, args.project_name, args.model, args.path_to_videos)
