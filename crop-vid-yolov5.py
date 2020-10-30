# import face_alignment
from argparse import ArgumentParser
from skimage import img_as_ubyte
from skimage.transform import resize
import imageio
import numpy as np
import warnings
from os import path
import sys
sys.path.append(path.abspath('../AnimAI-tion/yolov5'))

from scenedetect.platform import tqdm, get_and_create_path
from scene_detect_utils import find_scenes, get_end_frame_for_each_scene
import cv2

from yolov5.yolov5_model_inf import load_model, extract_bbox_from_frame

warnings.filterwarnings("ignore")

# example run
# CUDA_VISIBLE_DEVICES=0 python3 crop-video-testing.py --inp data/kon_(num).mkv --min_frames 60 --out /sizigi/coruscant/pond/data_proc/fom_ray/clips/kon(num) --name kon(num)

weights = 'weights/yolov5_small.pt'

# for a singular frame get all bboxes
def extract_bbox(frame, fa, model, device, half):
    if max(frame.shape[0], frame.shape[1]) > 640:
        scale_factor = max(frame.shape[0], frame.shape[1]) / 640.0
        frame = resize(frame, (int(frame.shape[0] / scale_factor), int(frame.shape[1] / scale_factor)))
        frame = img_as_ubyte(frame)
    else:
        scale_factor = 1
    frame = frame[..., :3]
    if fa is None:
        bboxes = list(extract_bbox_from_frame(frame, model, device, half))
    else:
        bboxes = fa.face_detector.detect_from_image(frame[..., ::-1])

    if len(bboxes) == 0:
        return []
    return np.array(bboxes)[:, :-1] * scale_factor


def bb_intersection_over_union(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou


def join(tube_bbox, bbox):
    xA = min(tube_bbox[0], bbox[0])
    yA = min(tube_bbox[1], bbox[1])
    xB = max(tube_bbox[2], bbox[2])
    yB = max(tube_bbox[3], bbox[3])
    return (xA, yA, xB, yB)


def compute_bbox(start, end, fps, tube_bbox, frame_shape, inp, image_shape, increase_area=0.1):
    left, top, right, bot = tube_bbox
    width = right - left
    height = bot - top

    # Computing aspect preserving bbox
    width_increase = max(increase_area, ((1 + 2 * increase_area) * height - width) / (2 * width))
    height_increase = max(increase_area, ((1 + 2 * increase_area) * width - height) / (2 * height))

    left = int(left - width_increase * width)
    top = int(top - height_increase * height)
    right = int(right + width_increase * width)
    bot = int(bot + height_increase * height)

    top, bot, left, right = max(0, top), min(bot, frame_shape[0]), max(0, left), min(right, frame_shape[1])
    h, w = bot - top, right - left

    start = start / fps
    end = end / fps
    time = end - start

    scale = f'{image_shape[0]}:{image_shape[1]}'

    return f'ffmpeg -i {inp} -ss {start} -t {time} -filter:v "crop={w}:{h}:{left}:{top}, scale={scale}"'


# gets bbox for each frame in a trajectory and saves each frame cropped by the bbox
def save_bbox_to_file(start, end, fps, tube_bbox, frame_shape, cur_traj, inp, image_shape,
                      vid_reader, increase_area=0.1, aspect_rat=True, pan_thresh=None, fa=None, rm_low_res=None,
                      model=None, device=None, half=None):

    left, top, right, bot = tube_bbox
    width = right - left
    height = bot - top

    # Computing aspect preserving bbox
    width_increase = max(increase_area, ((1 + 2 * increase_area) * height - width) / (2 * width))
    height_increase = max(increase_area, ((1 + 2 * increase_area) * width - height) / (2 * height))

    left = int(left - width_increase * width)
    top = int(top - height_increase * height)
    right = int(right + width_increase * width)
    bot = int(bot + height_increase * height)

    top, bot, left, right = max(0, top), min(bot, frame_shape[0]), max(0, left), min(right, frame_shape[1])
    h, w = bot - top, right - left

    # making sure that h and w are the same
    if aspect_rat:
        if h != w:
            # change right and left boundaries of box if height is greater than width
            if h < w:
                dif = w - h
                if dif % 2 == 0:
                    right -= int(dif / 2)
                    left += int(dif / 2)
                else:
                    right -= (dif // 2) + 1
                    left += dif // 2
                h, w = bot - top, right - left
            # change top and bottom boundaries of box if width is greater than height,
            else:
                dif = h - w
                if dif % 2 == 0:
                    bot -= int(dif / 2)
                    top += int(dif / 2)
                else:
                    bot -= (dif // 2) + 1
                    top += dif // 2

                h, w = bot - top, right - left

    if rm_low_res:
        if h < rm_low_res and w < rm_low_res:
            return

    if pan_thresh:
        end_frame = vid_reader.get_data(end)
        start_frame = vid_reader.get_data(start)
        end_bbox = extract_bbox(end_frame, fa, model, device, half)
        start_bbox = extract_bbox(start_frame, fa, model, device, half)


        # get bboxes closest to tube bbox (that represents the bbox for the current trajetory)
        if len(start_bbox) == 0:
            return
        if len(start_bbox) == 1:
            start_bbox = start_bbox[0]
        else:
            max_iou = 0
            bbox_num = 0
            for i in range(len(start_bbox)):
                cur_iou = bb_intersection_over_union(tube_bbox, start_bbox[i])
                if cur_iou > max_iou:
                    max_iou = cur_iou
                    bbox_num = i
            start_bbox = start_bbox[bbox_num]
            print('         TUBE START', start_bbox)
        # same thing for ending frame
        if len(end_bbox) == 0:
            return
        if len(end_bbox) == 1:
            end_bbox = end_bbox[0]
        else:
            max_iou = 0
            bbox_num = 0
            for i in range(len(end_bbox)):
                cur_iou = bb_intersection_over_union(tube_bbox, end_bbox[i])
                if cur_iou < max_iou:
                    max_iou = cur_iou
                    bbox_num = i
            end_bbox = end_bbox[bbox_num]
            print('         TUBE END', end_bbox)

        if bb_intersection_over_union(end_bbox, start_bbox) < pan_thresh:
            return

        # end_start_iou = bb_intersection_over_union(end_bbox, start_bbox)
        # if end_start_iou > pan_thresh:
        #     return

    for frame in range(start, end):
        cur_frame = vid_reader.get_data(frame)
        # cropped_frame = cur_frame[top:bot, left:right, :]
        cropped_frame = cv2.cvtColor(cur_frame[top:bot, left:right, :], cv2.COLOR_RGB2BGR)
        if h > image_shape[0] and w > image_shape[1]:
            resized_frame = cv2.resize(cropped_frame, dsize=(image_shape[0], image_shape[1]),
                                       interpolation=cv2.INTER_AREA)
        else:
            resized_frame = cv2.resize(cropped_frame, dsize=(image_shape[0], image_shape[1]),
                                       interpolation=cv2.INTER_CUBIC)

        file_name = '{}-traj{}-frame{}.png'.format(args.name, cur_traj, frame - start + 1)
        cv2.imwrite(get_and_create_path(file_name, args.out + '/' + args.name + '_traj' + str(cur_traj)),
                    resized_frame)


# runs save_bbox_to_file for every trajectory in a given clip
def save_bbox_traj_to_file(trajectories, fps, frame_shape, vid_reader, cur_traj, args,
                           aspect_rat=True, pan_thresh=None, fa=None, rm_low_res=None,
                           model=None, device=None, half=None):
    scene_traj = 0
    for i, (bbox, tube_bbox, start, end) in enumerate(trajectories):
        if (end - start) > args.min_frames:
            scene_traj += 1
            print('traj num {}'.format(cur_traj + scene_traj))
            save_bbox_to_file(start, end, fps, tube_bbox, frame_shape, cur_traj=cur_traj + scene_traj, inp=args.inp,
                              image_shape=args.image_shape, vid_reader=vid_reader, increase_area=args.increase,
                              aspect_rat=aspect_rat, pan_thresh=pan_thresh, fa=fa, rm_low_res=rm_low_res,
                              model=model, device=device, half=half)
    return scene_traj


def compute_bbox_trajectories(trajectories, fps, frame_shape, args):
    commands = []
    for i, (bbox, tube_bbox, start, end) in enumerate(trajectories):
        if (end - start) > args.min_frames:
            command = compute_bbox(start, end, fps, tube_bbox, frame_shape, inp=args.inp, image_shape=args.image_shape,
                                   increase_area=args.increase)
            commands.append(command)
    return commands


# def process_video(args, out_file, scene_changes):
def process_video(args, scene_changes, crop_output='file', out_file=None, aspect_rat=True, pan_thresh=None,
                  rm_low_res=128):
    device = 'cpu' if args.cpu else 'cuda'
    # fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=False,
    #                                   device=device, face_detector='blazeface')
    fa = None

    model, device, half = load_model(weights)

    video = imageio.get_reader(args.inp)

    trajectories = []
    previous_frame = None
    fps = video.get_meta_data()['fps']
    commands = []
    num = 0
    cur_traj = 0
    try:
        # loop for each frame and i is frame num
        for i, frame in tqdm(enumerate(video)):
            # frame is matrix of pixels
            frame_shape = frame.shape
            # extract_bbox returns bboxes
            # bbox in this format: [bbox_num, [crop_x, crop_y, crop_x2, crop_y2, score]]
            bboxes = extract_bbox(frame, fa, model, device, half)
            # For each trajectory check the criterion
            not_valid_trajectories = []
            valid_trajectories = []

            # trajectory in this format: (bbox, tube_bbox, start frame, end frame)
            for trajectory in trajectories:
                # new tube_bbox is old bbox
                tube_bbox = trajectory[0]
                if trajectory[3] in scene_changes:
                    # ending a trajectory if the frame is the frame of a scene change
                    not_valid_trajectories.append(trajectory)
                    continue

                intersection = 0
                # bboxes in this format: [bbox_num, [crop_x, crop_y, crop_x2, crop_y2, score]]
                for bbox in bboxes:
                    intersection = max(intersection, bb_intersection_over_union(tube_bbox, bbox))
                if intersection > args.iou_with_initial:
                    valid_trajectories.append(trajectory)
                else:
                    not_valid_trajectories.append(trajectory)

            if crop_output == "file":
                cur_traj += save_bbox_traj_to_file(not_valid_trajectories, fps, frame_shape, video, cur_traj,
                                                   args, aspect_rat, pan_thresh=pan_thresh, fa=fa,
                                                   rm_low_res=rm_low_res, model=model, device=device,
                                                   half=half)
            elif crop_output == 'ffmpeg cmd':
                # get ffmpeg commands to clip and crop vid for not valid trajectories,
                # bbox is valid up until this point, so get command for it
                new_cmd = compute_bbox_trajectories(not_valid_trajectories, fps, frame_shape, args)
                for cmd in new_cmd:
                    commands.append(cmd)
                    out_file.write("{} {}_{}.mp4\n".format(cmd, args.out_crops, num))
                    out_file.flush()
                    num += 1

            # continue on with valid trajectories
            trajectories = valid_trajectories

            ## Assign bbox to trajectories, create new trajectories
            for bbox in bboxes:
                intersection = 0
                current_trajectory = None
                # trajectory in this format: (bbox, tube_bbox, start frame, end frame)
                for trajectory in trajectories:
                    # new tube_bbox is old bbox
                    tube_bbox = trajectory[0]
                    # current_intersection is iou
                    current_intersection = bb_intersection_over_union(tube_bbox, bbox)
                    if intersection < current_intersection and current_intersection > args.iou_with_initial:
                        intersection = bb_intersection_over_union(tube_bbox, bbox)
                        current_trajectory = trajectory

                ## Create new trajectory
                if current_trajectory is None:
                    # initial trajectory
                    trajectories.append([bbox, bbox, i, i])
                else:
                    # change end frame to current frame
                    current_trajectory[3] = i
                    # join function combines tube_bbox and bbox to output new tube_bbox
                    current_trajectory[1] = join(current_trajectory[1], bbox)


    except IndexError as e:
        raise (e)

    if crop_output == 'ffmpeg cmd':
        new_cmd = compute_bbox_trajectories(trajectories, fps, frame_shape, args)
        for cmd in new_cmd:
            commands.append(cmd)
            out_file.write("{} {}_{}.mp4\n".format(cmd, args.out_crops, num))
            out_file.flush()
            num += 1
        return commands


if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument("--image_shape", default=(256, 256), type=lambda x: tuple(map(int, x.split(','))),
                        help="Image shape")
    parser.add_argument("--increase", default=0.1, type=float, help='Increase bbox by this amount')
    parser.add_argument("--iou_with_initial", type=float, default=0.25, help="The minimal allowed iou with inital bbox")
    parser.add_argument("--inp", required=True, help='Input image or video')
    parser.add_argument("--min_frames", type=int, default=150, help='Minimum number of frames')
    parser.add_argument("--cpu", dest="cpu", action="store_true", help="cpu mode.")
    parser.add_argument("--out", default=None, help='where to write output crops txt')
    parser.add_argument("--out_crops", default="", help='where to write output crops in ffmpeg command')
    parser.add_argument("--name", required=True, help='name of video being clipped')
    parser.add_argument("--pan_thresh", type=float,
                        help='threshold of iou difference between end and start frame of a trajectory')
    parser.add_argument("--rm_low_res", type=int,
                        help='minimum threshold of square pixel size for trajectory to be saved')

    args = parser.parse_args()

    # getting list of the ending frame for each scene
    scenes = find_scenes(args.inp)
    l_scene_end = get_end_frame_for_each_scene(scenes)
    process_video(args, l_scene_end, pan_thresh=args.pan_thresh)

    # cmd line run
    # CUDA_VISIBLE_DEVICES=0 python3 crop-vid-yolov5.py --inp data_to_crop/kon_cropped1.mp4 --min_frames 60 --out data_cropped --name kon1_cropped
    # CUDA_VISIBLE_DEVICES=0 python3 crop-vid-yolov5.py --inp data_to_crop/kon_cropped1.mp4 --min_frames 60 --out data_cropped --name kon1_cropped --pan_thresh .25