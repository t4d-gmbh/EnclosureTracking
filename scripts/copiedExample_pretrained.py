import deeplabcut as dlc
import os

# Note: we should use a __main__ logic here!

# Setting some parameters (ideally we get this from argparse)
project_path = "/content/demo-me-2021-07-14"
config_path = os.path.join(project_path, "config.yaml")
video = os.path.join(project_path, "videos", "videocompressed1.mp4")

# Detect the animals in the frames
dlc.analyze_videos(config_path,[video], shuffle=0, videotype="mp4",auto_track=False)

# Prepare for tracking
TRACK_METHOD = "ellipse"  # Could also be "box", but "ellipse" was found to be more robust on this dataset.

dlc.convert_detections2tracklets(
    config_path,
    [video],
    videotype='mp4',
    shuffle=0,
    track_method=TRACK_METHOD,
    ignore_bodyparts=["tail1", "tail2", "tailend"],  # Some body parts can optionally be ignored during tracking for better assembly (but they are used later)
)

# Combine the frames
dlc.stitch_tracklets(
    config_path,
    [video],
    videotype='mp4',
    shuffle=0,
    track_method=TRACK_METHOD,
    n_tracks=3,
)

#Filter the predictions to remove small jitter, if desired:
dlc.filterpredictions(config_path, 
                                 [video], 
                                 shuffle=0,
                                 videotype='mp4', 
                                 track_method = TRACK_METHOD)

dlc.create_labeled_video(
    config_path,
    [video],
    videotype='mp4',
    shuffle=0,
    color_by="individual",
    keypoints_only=False,
    draw_skeleton=True,
    filtered=True,
    track_method=TRACK_METHOD,
)

dlc.plot_trajectories(config_path, [video], shuffle=0,videotype='mp4', track_method=TRACK_METHOD)
