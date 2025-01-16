"""Collection of helpful functions and definitions"""
import os
import yaml
import glob
import warnings
from pathlib import Path

body_parts = ["nose",
"left_ear",
"right_ear",
"left_ear_tip",
"right_ear_tip",
"left_eye",
"right_eye",
"neck",
"mid_back",
"mouse_center",
"mid_backend",
"mid_backend2",
"mid_backend3",
"tail_base",
"tail1",
"tail2",
"tail3",
"tail4",
"tail5",
"left_shoulder",
"left_midside",
"left_hip",
"right_shoulder",
"right_midside",
"right_hip",
"tail_end",
"head_midpoint"]

parts_mapping = {part:part for part in body_parts}

skeleton_layout = [
    ['nose', 'head_midpoint'],
    ['left_ear', 'head_midpoint'],
    ['left_ear_tip', 'left_ear'],
    ['right_ear', 'head_midpoint'],
    ['right_ear_tip', 'right_ear'],
    ['head_midpoint', 'left_eye'],
    ['head_midpoint', 'right_eye'],
    ['head_midpoint', 'neck'],
    ['neck', 'mid_back'],
    ['mid_back', 'mouse_center'],
    ['mid_back', 'mid_backend2'],
    ['mid_backend2', 'mid_backend3'],
    ['mid_backend3', 'tail_base'],
    ['tail_base', 'tail1'],
    ['tail1', 'tail2'],
    ['tail2', 'tail3'],
    ['tail3', 'tail4'],
    ['tail4', 'tail5'],
    ['tail5', 'tail_end'],
    ['left_shoulder', 'mid_back'],
    ['left_midside', 'mouse_center'],
    ['left_hip', 'mid_backend3'],
    ['right_shoulder', 'mid_back'],
    ['right_midside', 'mouse_center'],
    ['right_hip', 'mid_backend3'],
    ['head_midpoint', 'neck'],
    ['head_midpoint', 'nose']
]


def to_pretrained_multianimal(config_file:str|Path, nbr_animals:int=10,
                              output_file:str|Path|None=None, **config_params)->str|Path:
    """Converts an existing DLC project config file to a multi-animal project

    Parameters
    ----------
    config_file:
      Path to the existing project config file.
    nbr_animals:
      The number of animals that are maximally visible together
    output_file:
      Optional location to export the adapted config file to.
      If unset (i.e. `None`) then the provided `config_file` is overwritten.
    **config_params:
      Further, optional keywords can be set.
      E.g. `config_params={'default_net_type': 'dlcrnet_ms5'}`
    
    Returns
    -------
      config_file:
        Path the the updated/new configuration file
    """
    # Step 1: Read the original YAML file
    with open(config_file, 'r') as file:
        data = yaml.safe_load(file)
    
    # Step 2: Modify the data
    data['Task'] = 'Pretrained'  # Change Task
    # data['scorer'] = 'Jonas'  # Change scorer
    data['multianimalproject'] = True  # Change multianimalproject
    data['identity'] = False  # Change identity
    # data['project_path'] = '/home/jonas/Projects/AdaptingTopViewMouse/Pretrained-Jonas-2025-01-15'  # Update project path
    
    # Add individuals
    data['individuals'] = [f"individual{i}" for i in range(1, nbr_animals + 1)]
    
    # Add unique and multianimal bodyparts
    data['uniquebodyparts'] = []
    data['multianimalbodyparts'] = data['bodyparts']  # Assuming you want to keep the same bodyparts
    
    # Update bodyparts
    data['bodyparts'] = 'MULTI!'
    
    # Update skeleton
    data['skeleton'] = skeleton_layout    

    # Update Training configuration
    # data['default_net_type'] = 'dlcrnet_ms5'
    data['default_augmenter'] = 'multi-animal-imgaug'
    data['default_track_method'] = 'ellipse'

    # Set all further configs
    for k,v in config_params.items():
        data[k] = v
    
    # Step 3: Write the modified data back to the YAML file
    _out_file = output_file or config_file
    with open(_out_file, 'w') as file:
        yaml.dump(data, file, default_flow_style=False)

    return _out_file

def get_config_path(working_dir:str, project_name: str|None,
                    user:str|None, date:str|None=None):
    """Retrieve the path to a project config file.

    Basically fight against the automated name creation form DLC...

    Parameters
    ----------
    working_dir:
        The path to the location/folder the preject resides in
    project_name:
        The project name.
        If set to `None` then a wildchard search will try to find the project.
    user:
        Value of the `experimenter` argument from dlc.create_pretrained_project
        If set to `None` then a wildchard search will try to find the project.
    date:
        Optional date sring (format YYYY-MM-DD) when the project was initiated
    """
    config_path = None
    project_path = f"{project_name or '*'}-{user or '*'}-{date or '*'}"
    for path in glob.glob(os.path.join(working_dir, project_path)):
        found_path = False
        if os.path.isdir(path):
            if found_path:
                warnings.warn(
                    f"Found multiple locations for {project_name=} and {user=}.\n"
                    f"The first match is used: {config_path=}"
                )
                break
            config_path = os.path.join(path, 'config.yaml')
            found_path = True
    assert config_path is not None
    return config_path
