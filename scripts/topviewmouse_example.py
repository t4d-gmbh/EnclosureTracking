import deeplabcut as dlc


def main(video_path:str, superanimal_name:str, scale_list) -> None:
    """Use ModelZoo to detect poses

    Parameters
    ----------
    video_path:
      Path to the vide ot analyze
    superanimal_name:
      Identifer of the pretrained model to use
    scale_list:
      A collectio of mouse sizes (in # pixels) to test
    """

    dlc.video_inference_superanimal([video_path],
                                    superanimal_name,
                                    scale_list=scale_list,
                                    video_adapt = False)
    return None

if __name__ == '__main__':
    video_path = 'demo-video.mp4'
    superanimal_name = 'superanimal_quadruped_dlcrnet'

    scale_list = range(50, 200, 50)

    main(video_path=video_path,
         superanimal_name=superanimal_name,
         scale_list=scale_list)
