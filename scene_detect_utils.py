from argparse import ArgumentParser
from scenedetect import VideoManager, SceneManager, video_splitter
from scenedetect.detectors import ContentDetector
from scenedetect.platform import get_cv2_imwrite_params, tqdm, get_and_create_path
from imageio import get_reader
# from gen_images import gen_img_folders
# import matplotlib.pyplot as plt
import cv2

# example run
# python3 scene_detect_simple.py --inp data/kon_cut1.mp4 --out data/scene_det --name kon

# threshold of differences between scenes to determine whether or not a frame change counts as a scene change
# threshold calculated by change of average HSV (hue, saturation, and luminance)
def find_scenes(video_path, threshold=30.0):
    # Create our video & scene managers, then add the detector.
    video_manager = VideoManager([video_path])
    scene_manager = SceneManager()
    scene_manager.add_detector(
        ContentDetector(threshold=threshold))

    # Base timestamp at frame 0 (required to obtain the scene list).
    base_timecode = video_manager.get_base_timecode()

    # Improve processing speed by downscaling before processing.
    video_manager.set_downscale_factor()

    # Start the video manager and perform the scene detection.
    video_manager.start()
    scene_manager.detect_scenes(frame_source=video_manager)

    # Each returned scene is a tuple of the (start, end) timecode.
    return scene_manager.get_scene_list(base_timecode)


def get_end_frame_for_each_scene(scenes):
    l_scene_end = []
    for i in range(len(scenes)):
        l_scene_end.append(scenes[i][1].frame_num)
    return l_scene_end


def write_frames_of_scenes(video_path, output_dir, vid_name, l_scene_end):
    vid_read = get_reader(video_path)
    curr_scene = 1
    frame_num = 0

    for i, frame in tqdm(enumerate(vid_read)):
        if i in l_scene_end:
            curr_scene += 1
            frame_num = 0
        frame_num += 1
        file_name = '{}-scene{}-frame{}.png'.format(vid_name, curr_scene, frame_num)
        cv2.imwrite(get_and_create_path(file_name, output_dir + '/scene' + str(curr_scene)),
                    cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

    print('Completed writing files')


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--inp", required=True, help='Input image or video')
    parser.add_argument("--name", required=True, help='name of video being clipped')
    parser.add_argument("--out", required=True, help='where to write output clips')
    args = parser.parse_args()

    video_path = args.inp
    output_dir = args.out
    vid_name = args.name

    # video_path = 'data/kon_cut1.mp4'
    # output_dir = 'data/scene_det'
    # vid_name = 'kon'

    scenes = find_scenes(video_path)
    l_scene_end = get_end_frame_for_each_scene(scenes)
    write_frames_of_scenes(video_path, output_dir, vid_name, l_scene_end)
