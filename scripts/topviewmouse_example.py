import argparse
import deeplabcut as dlc


def main(video_path: str, superanimal_name: str, scale_list) -> None:
    """Use ModelZoo to detect poses

    Parameters
    ----------
    video_path:
      Path to the video to analyze
    superanimal_name:
      Identifier of the pretrained model to use
    scale_list:
      A collection of mouse sizes (in # pixels) to test
    """

    dlc.video_inference_superanimal([video_path],
                                    superanimal_name,
                                    scale_list=scale_list,
                                    video_adapt=False)
    return None


if __name__ == '__main__':
    # Set up argument parsing
    parser = argparse.ArgumentParser(description='Analyze video using DeepLabCut ModelZoo.')
    parser.add_argument('video_path', type=str, help='Path to the video to analyze')
    parser.add_argument('superanimal_name', type=str, help='Identifier of the pretrained model to use')
    parser.add_argument('--scale_list', type=int, nargs='+', default=list(range(50, 200, 50)),
                        help='A collection of mouse sizes (in # pixels) to test (default: 50 100 150)')

    # Parse the arguments
    args = parser.parse_args()

    # Call the main function with parsed arguments
    main(video_path=args.video_path,
         superanimal_name=args.superanimal_name,
         scale_list=args.scale_list)
