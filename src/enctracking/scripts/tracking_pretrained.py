"""This script performs tracking of individuals in a video.

This is script 4 in the workflow.

.. note::
    Running this only really makes sense if the pose estimation results are not too bad.
    See/run [evaluate_pretrained.py](./evaluate_pretrained.py) for details.
"""
from typing import Collection

import argparse
import deeplabcut as dlc

from ..helpers import (
    get_config_path,
)

def tracking_pretrained(user:str, working_dir:str, project_name:str, videos_to_analyze:Collection, batch_size:int):
    """Analyze videos to track individuals using a trained DeepLabCut model.

    This function performs the following steps:
    1. Retrieves the configuration path for the specified project.
    2. Analyzes the specified videos to track individuals.
    3. Creates annotated videos to check the performance of the tracking.

    Args:
        user (str): The username of the experimenter for the project.
        working_dir (str): The directory where the project is located.
        project_name (str): The name of the existing project.
        videos_to_analyze (list of str): A list of video file paths to be analyzed.
        batch_size (int): The batch size to use when fine-tuning.

    Returns:
        None: This function does not return any value. It performs actions to analyze
        the videos and create annotated outputs.
    
    Raises:
        Exception: If there is an error during the analysis of the videos.
    """
    config_path = get_config_path(working_dir=working_dir,
                                  project_name=project_name,
                                  user=user)

    # Run the analysis of the videos
    scorername = dlc.analyze_videos(config=config_path,
                                     videos=videos_to_analyze,
                                     batch_size=batch_size)

    # Create some annotated videos to check the performance
    dlc.create_video_with_all_detections(config=config_path,
                                          videos=videos_to_analyze)

def get_args():
    """Fetch command line arguments
    """
    parser = argparse.ArgumentParser(
        description="Perform tracking of individuals in a video using a trained DeepLabCut model.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument('--user', type=str, default='ml_user', help='Username for the project.')
    parser.add_argument('--working_dir', type=str, default='/home/ml_user', help='Working directory for the project.')
    parser.add_argument('--project_name', type=str, default='PretrainedTracker', help='Name of the existing project.')
    parser.add_argument('--videos_to_analyze', type=str, nargs='+', required=True, help='List of video files to analyze.')
    parser.add_argument('--batch_size', type=int, default=2, help='The batch size to use when fine-tuning.')

    # Add example usage to the help message
    parser.epilog = (
        "Example usage:\n"
        "  tracking_pretrained --user new_user --working_dir /home/new_user --project_name NewTracker "
        "--videos_to_analyze /path/to/video1.mp4 /path/to/video2.mp4 --batch_size 8\n"
        "  tracking_pretrained --videos_to_analyze /path/to/video1.mp4  # Use default values for other parameters"
    )

    args = parser.parse_args()
    
    return args

def main():
    """Script entrypoint
    """
    args = get_args()

    tracking_pretrained(user=args.user, working_dir=args.working_dir,
                        project_name=args.project_name,
                        videos_to_analyze=args.videos_to_analyze,
                        batch_size=args.batch_size)

if __name__ == "__main__":
    main()
