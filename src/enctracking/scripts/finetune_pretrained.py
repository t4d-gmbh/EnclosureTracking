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
    """Create a DeepLabCut project, label images, create a training dataset, and train the network.

    This function performs the following steps:
    1. Creates a new DeepLabCut project with a pre-trained model.
    2. Labels images using the GUI (this step is recommended to be done manually).
    3. Converts the labeled data to the required format.
    4. Checks the labels for correctness.
    5. Creates a training dataset by mapping markers.
    6. Trains the network using the specified parameters.

    Args:
        user (str): The username of the experimenter for the project.
        working_dir (str): The directory where the project will be created.
        project_name (str): The name of the project to be created.
        model (str): The name of the pretrained model to be used.
        path_to_videos (str): The path to the video files that will be used for training.

    Returns:
        None: This function does not return any value. It performs actions to create
        and configure the DeepLabCut project and train the network.
    
    Raises:
        Exception: If there is an error during any of the steps in the process.
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

    # Label images (this step is best done using the GUI)
    # NOTE: This is best done using the GUI.
    #       Simply open a new terminal and run:
    #       `ipython -m deeplabcut`.
    #       Then load the project (i.e. open the
    #       config.yaml and go to the panel 'label data'
    #       (or similar)
    
    # Convert CSV to H5 format
    dlc.convertcsv2h5(config_path, scorer=user)
    
    # Check the labels
    dlc.check_labels(config_path, visualizeindividuals=True)

    # Create the training dataset
    dlc.modelzoo.utils.create_conversion_table(config=config_path,
                                               super_animal=model,
                                               project_to_super_animal=parts_mapping) 
    
    # Create the actual dataset
    dlc.create_multianimaltraining_dataset(config_path, net_type='dlcrnet_ms5',
                                           detector_type='fasterrcnn_mobilenet_v3_large_fpn')

    # Train the network (fine-tuning)
    torch_params = dict(
        batch_size=4,
    )
    dlc.train_network(config=config_path, epochs=None, **torch_params)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Posture tracking with a pretrained model and fine-tuning.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument('--user', type=str, default='ml_user', help='Username for the project.')
    parser.add_argument('--working_dir', type=str, default='/home/ml_user', help='Working directory for the project.')
    parser.add_argument('--project_name', type=str, default='PretrainedTracker', help='Name of the project.')
    parser.add_argument('--model', type=str, default='superanimal_topviewmouse', help='Pretrained model to use.')
    parser.add_argument('--path_to_videos', type=str, default='/home/ml_user/data/samples', help='Path to the videos.')

    # Add example usage to the help message
    parser.epilog = (
        "Example usage:\n"
        "  python your_script.py --user new_user --working_dir /home/new_user --project_name NewTracker "
        "--model another_model --path_to_videos /path/to/videos\n"
       
