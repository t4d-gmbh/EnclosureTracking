"""This script documents the necessary steps to carry out posture tracking
with a pretrained model and fine-tuning.

This tries to follow:
https://deeplabcut.github.io/DeepLabCut/docs/maDLC_UserGuide.html
"""

import deeplabcut as dlc
import argparse

from ..helpers import (
    parts_mapping,
    get_config_path,
)

def finetune_pretrained(user:str, working_dir:str, project_name:str, model:str,
                        batch_size:int):
    """Create a DeepLabCut project, label images, create a training dataset, and train the network.

    This function performs the following steps:
    1. Converts labeled data to the required format.
    2. Checks the labels for correctness.
    3. Creates a training dataset by mapping markers.
    4. Trains the network using the specified parameters.

    Args:
        user (str): The username of the experimenter for the project.
        working_dir (str): The directory where the project will be created.
        project_name (str): The name of the project to be created.
        model (str): The name of the pretrained model to be used.
        batch_size (int): The batch size to use when fine-tuning.

    Returns:
        None: This function does not return any value. It performs actions to create
        and configure the DeepLabCut project and train the network.
    
    Raises:
        Exception: If there is an error during any of the steps in the process.
    """
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
    # NOTE: This step is needed if the labeling was carried out
    #       by some other user.
    #       Unfortunately, this step is not sufficient to include/
    #       use data labeled by others.
    #       The additional steps (priort to running this) are:
    #       - Remove the *.h5 file in each fo the video folders
    #       - Rename the CollectedData_<otheruser>.csv to
    #         CollectedData_<user>.csv
    #       - Replace all occurencens of <otheruser> with <user>
    #       - in each of the CollectedData_<user>.csv files
    #       - Finally, run this scipt (or just dlc.convertcsv2h5)
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
        batch_size=batch_size,
    )
    dlc.train_network(config=config_path, epochs=None, **torch_params)

def get_args():
    """Fetch command line arguments
    """

    parser = argparse.ArgumentParser(
        description="Posture tracking with a pretrained model and fine-tuning.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument('--user', type=str, default='ml_user', help='Username for the project.')
    parser.add_argument('--working_dir', type=str, default='/home/ml_user', help='Working directory for the project.')
    parser.add_argument('--project_name', type=str, default='PretrainedTracker', help='Name of the project.')
    parser.add_argument('--model', type=str, default='superanimal_topviewmouse', help='Pretrained model to use.')
    parser.add_argument('--batch_size', type=int, default=2, help='The batch size to use when fine-tuning.')

    # Add example usage to the help message
    parser.epilog = (
        "Example usage:\n"
        "  finetune_pretrained --user new_user --working_dir /home/new_user --project_name NewTracker "
        "--model superanimal_topviewmouse --batch_size 8\n"
    )

    args = parser.parse_args()

    return args

def main():
    """Script entrypoint
    """
    args = get_args()
    finetune_pretrained(
        user=args.user, working_dir=args.working_dir,
        project_name=args.project_name, model=args.model,
        batch_size=args.batch_size
    )

if __name__ == "__main__":
    main()
