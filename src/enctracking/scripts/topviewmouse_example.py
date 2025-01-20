import warnings
import argparse
from pathlib import Path
from deeplabcut.modelzoo.video_inference import video_inference_superanimal

def main(video_path: str,
         dest_folder: str,
         superanimal_name: str,
         model_name: str,
         detector_name: str,
         max_individuals: int,
         device: str,
         **kwargs
         ) -> None:
    """Use ModelZoo to detect poses

    Parameters
    ----------
    video_path:
      Path to the video to analyze
    superanimal_name:
      Identifier of the pretrained model to use
    """
    out_dir = Path(dest_folder)
    if not out_dir.exists():
        warnings.warn(f"Creating new directory {str(out_dir)}")
        out_dir.mkdir(parents=True)

    video_inference_superanimal(
        videos=[video_path],
        superanimal_name=superanimal_name,
        model_name=model_name,
        detector_name=detector_name,
        max_individuals=max_individuals,
        device=device,
        **kwargs
        )
    return None


if __name__ == '__main__':
    # Set up argument parsing
    parser = argparse.ArgumentParser(description='Analyze video using DeepLabCut ModelZoo.')
    parser.add_argument('video_path', type=str, help='Path to the video to analyze')
    parser.add_argument('--dest_folder', type=str,
                        help='Where to store the results')
    parser.add_argument('--superanimal_name', type=str,
                        default="superanimal_topviewmouse",
                        help='Identifier of the pretrained model to use')
    parser.add_argument('--model_name', type=str,
                        default="hrnet_w32",
                        help='What network model to use')
    parser.add_argument('--detector_name', type=str,
                        default="fasterrcnn_resnet50_fpn_v2",
                        help='What model to use to detect animals')
    parser.add_argument('--max_individuals', type=int,
                        default=3,
                        help='How many different individuals are maximally visible')
    parser.add_argument('--device', type=str,
                        default='auto',
                        help='What device to use')
    # Parse the arguments
    args = parser.parse_args()

    # setting all parameters
    params = dict(
        detector_batch_size=4,
        video_adapt=True,
        batch_size=4,
        video_adapt_batch_size=4,
        pseudo_threshold=0.05,
        bbox_threshold=0.9,
        detector_epochs=8,
        pose_epochs=8,
    )
    
    # echo params
    print(f"{args=}")
    for k,v in params.items():
        print(f"{k}={v}")


    # Call the main function with parsed arguments
    main(video_path=args.video_path,
         device=args.device,
         dest_folder=args.dest_folder,
         superanimal_name=args.superanimal_name,
         model_name=args.model_name,
         detector_name=args.detector_name,
         max_individuals=args.max_individuals,
         **params
         )
