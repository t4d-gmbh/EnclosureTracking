"""This script adds new video files to an existing project.

This is an optional second step after running the init_pretrained.py script.
"""

import deeplabcut as dlc
import argparse

from ..helpers import (
    get_config_path,
)

def add_videos(user, working_dir, project_name, videos_to_add):
    """Add new video files to an existing DeepLabCut project.

    This function performs the following steps:
    1. Retrieves the configuration path for the specified project.
    2. Adds the specified video files to the existing project.

    Args:
        user (str): The username of the experimenter for the project.
        working_dir (str): The directory where the project is located.
        project_name (str): The name of the existing project.
        path_to_videos (str): The path to the directory containing the videos.
        videos_to_add (list of str): A list of video file paths to be added to the project.

    Returns:
        None: This function does not return any value. It performs actions to add
        the specified videos to the DeepLabCut project.
    
    Raises:
        Exception: If there is an error during the addition of videos to the project.
    """
    config_path = get_config_path(working_dir=working_dir,
                                  project_name=project_name,
                                  user=user)

    dlc.add_new_videos(config=config_path,
                       videos=videos_to_add)

def get_args():
    """Fetch command line arguments
    """
    parser = argparse.ArgumentParser(
        description="Add new video files to an existing DeepLabCut project.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument('--user', type=str, default='ml_user', help='Username for the project.')
    parser.add_argument('--working_dir', type=str, default='/home/ml_user', help='Working directory for the project.')
    parser.add_argument('--project_name', type=str, default='PretrainedTracker', help='Name of the existing project.')
    parser.add_argument('--videos_to_add', type=str, nargs='+', required=True, help='List of video files to add to the project.')

    # Add example usage to the help message
    parser.epilog = (
        "Example usage:\n"
        "  python your_script.py --user new_user --working_dir /home/new_user --project_name NewTracker "
        "--path_to_videos /path/to/videos --videos_to_add /path/to/video1.mp4 /path/to/video2.mp4\n"
        "  python your_script.py --videos_to_add /path/to/video1.mp4 /path/to/video2.mp4  # Use default values for other parameters"
    )

    args = parser.parse_args()
    
    return args

def main():
    """Script entrypoint
    """
    args = get_args()
    add_videos(user=args.user, working_dir=args.working_dir,
               project_name=args.project_name, videos_to_add=args.videos_to_add)

if __name__ == "__main__":
    main()
