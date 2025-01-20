"""Post-training steps to evaluate a trained model.

This is script 3 in the workflow.
"""

import deeplabcut as dlc
import argparse

from ..helpers import (
    parts_mapping,
    to_pretrained_multianimal,
    get_config_path,
)

def main(user, working_dir, project_name, model):
    """Evaluate a trained DeepLabCut model.

    This function performs the following steps:
    1. Retrieves the configuration path for the specified project.
    2. Evaluates the trained model using the specified configuration.

    Args:
        user (str): The username of the experimenter for the project.
        working_dir (str): The directory where the project is located.
        project_name (str): The name of the existing project.
        model (str): The name of the pretrained model to be evaluated.

    Returns:
        None: This function does not return any value. It performs actions to evaluate
        the trained model.
    
    Raises:
        Exception: If there is an error during the evaluation of the model.
    """
    config_path = get_config_path(working_dir=working_dir,
                                  project_name=project_name,
                                  user=user)

    dlc.evaluate_network(config=config_path, plotting=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Post-training steps to evaluate a trained DeepLabCut model.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument('--user', type=str, default='ml_user', help='Username for the project.')
    parser.add_argument('--working_dir', type=str, default='/home/ml_user', help='Working directory for the project.')
    parser.add_argument('--project_name', type=str, default='PretrainedTracker', help='Name of the existing project.')
    parser.add_argument('--model', type=str, default='superanimal_topviewmouse', help='Name of the pretrained model to evaluate.')

    # Add example usage to the help message
    parser.epilog = (
        "Example usage:\n"
        "  python your_script.py --user new_user --working_dir /home/new_user --project_name NewTracker --model another_model\n"
        "  python your_script.py  # Use default values for all parameters"
    )

    args = parser.parse_args()

    main(args.user, args.working_dir, args.project_name, args.model)
