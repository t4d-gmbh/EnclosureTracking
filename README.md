# Enclosure Tracking

Scripts to perform tracking and pose estimation in the outside enclosures.

This repository contains a series of scripts designed to facilitate posture tracking using DeepLabCut.
Each script corresponds to a specific step in the workflow, from project creation to model evaluation.

## Installation

To install the necessary package that contains these scripts, you can use `pip` to install directly from the GitHub repository:

```
pip install git+https://github.com/t4d-gmbh/EnclosureTracking.git@main
```

Once installed you can run any of the scripts explained below directly in the command line.

## Scripts Overview

### 1. `init_pretrained`
This script creates a new DeepLabCut project using a pretrained model and prepares it for labeling.

**Key Steps:**
- Create a project with a pretrained model.
- Convert the project into a multi-animal 

**Usage:**
```
init_pretrained --user <username> --working_dir <path> --project_name <project_name> --model <model_name> --path_to_videos <path_to_videos>
```

### 2. `add_videos`
This script adds new video files to an existing DeepLabCut project.

**Key Steps:**
- Retrieve the configuration path for the specified project.
- Add the specified video files to the existing project.

**Usage:**
```
add_videos --user <username> --working_dir <path> --project_name <project_name> --videos_to_add <video1> <video2> ...
```

### 3. `evaluate_pretrained`
This script evaluates a trained DeepLabCut model.

**Key Steps:**
- Retrieve the configuration path for the specified project.
- Evaluate the trained model using the specified configuration.

**Usage:**
```
evaluate_pretrained --user <username> --working_dir <path> --project_name <project_name>
```

### 4. `finetune_pretrained`
This script performs training (a fine-tuning) of the pretrained model.

**Key Steps:**
- Retrieve the configuration path for the specified project.
- Analyze the specified videos to track individuals.
- Create annotated videos to check the performance of the tracking.

**Usage:**
```
finetune_pretrained --user <username> --working_dir <path> --project_name <project_name> --model <model_name> --path_to_videos <path> --videos_to_analyze <video1> <video2> ...
```

### 5. `track_individuals.py`
This script performs tracking of individuals in a video.

**Key Steps:**
- Retrieve the configuration path for the specified project.
- Analyze the specified videos to track individuals.
- Create annotated videos to check the performance of the tracking.

**Usage:**
```
python -m enctracking.track_individuals --user <username> --working_dir <path> --project_name <project_name> --model <model_name> --path_to_videos <path> --videos_to_analyze <video1> <video2> ...
```

## Requirements
- DeepLabCut
- Python 3.x
- Required libraries as specified in the project.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
