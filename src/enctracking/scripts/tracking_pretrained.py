"""This script performs tracking of individuals in a video.

This is script 4 in the workflow.

.. note::
    Running this only really makes sense if the pose estimation results are not too bad.
    See/run [evaluate_pretrained.py](./evaluate_pretrained.py) for details.
"""

import os
import deeplabcut as dlc
import argparse

from ..helpers import (
    parts_mapping,
    to_pretrained_multianimal,
    get_config_path,
)

def main(user, working_dir, project_name, model, path_to_videos, videos_to_analyze):
    """Analyze videos to track individuals using a trained DeepLabCut model.

    This function performs the following steps:
    1. Retrieves the configuration path for the specified project.
    2. Analyzes the specified videos to track individuals.
    3. Creates annotated videos to check the performance of the tracking.

    Args:
        user (str): The username of the experimenter for the project.
        working_dir (str): The directory where the project is located.
        project_name (str): The name of the existing project.
        model (str): The name of the pretrained model to be used.
        path_to_videos (str): The path to the directory containing the videos.
        videos_to_analyze (list of str): A list of video file paths to be analyzed.

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
                                     batch_size=4)

    # Create some annotated videos to check the performance
    dlc.create_video_with_all_detections(config=config_path,
                                          videos=videos_to_analyze)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Perform tracking of individuals in a video using a trained DeepLabCut model.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument('--user', type=str, default='ml_user', help='Username for the project.')
    parser.add_argument('--working_dir', type=str, default='/home/ml_user', help='Working directory for the project.')
    parser.add_argument('--project_name', type=str, default='PretrainedTracker', help='Name of the existing project.')
    parser.add_argument('--model', type=str, default='superanimal_topviewmouse', help='Pretrained model to use.')
    parser.add_argument('--path_to_videos', type=str, default='/home/ml_user/data/samples', help='Path to the directory containing the videos.')
    parser.add_argument('--videos_to_analyze', type=str, nargs='+', required=True, help='List of video files to analyze.')

    # Add example usage to the help message
    parser.epilog = (
        "Example usage:\n"
        "  python your_script.py --user new_user --working_dir /home/new_user --project_name NewTracker "
        "--model another_model --path_to_videos /path/to/videos --videos_to_analyze /path/to/video1.mp4 /path/to/video2.mp4\n"
        "  python your_script.py --videos_to_analyze /path/to/video1.mp4  # Use default values for other parameters"
    )

    args = parser.parse_args()

    main(args.user, args.working_dir, args.project_name, args.model, args.path_to_videos, args.videos_to_analyze)
