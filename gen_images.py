from string import Template
from scenedetect.platform import get_cv2_imwrite_params, tqdm, get_and_create_path
import math
import cv2

def gen_img_folders(scene_list, video_manager, video_name, output_dir, num_images=2,
                    image_extension='jpg', quality_or_compression=95,
                    image_name_template='$VIDEO_NAME-Scene-$SCENE_NUMBER-$IMAGE_NUMBER',
                    downscale_factor=1, show_progress=False):
    # type: (...) -> bool
    """
    TODO: Documentation.
    Arguments:
        quality_or_compression: For image_extension=jpg or webp, represents encoding quality,
        from 0-100 (higher indicates better quality). For WebP, 100 indicates lossless.
        Default value in the CLI is 95 for JPEG, and 100 for WebP.
        If image_extension=png, represents the compression rate, from 0-9. Higher values
        produce smaller files but result in longer compression time. This setting does not
        affect image quality (lossless PNG), only file size. Default value in the CLI is 3.
        [default: 95]
    Returns:
        True if all requested images were generated & saved successfully, False otherwise.
    """

    if not scene_list:
        return True
    if num_images <= 0:
        raise ValueError()

    imwrite_param = []
    available_extensions = get_cv2_imwrite_params()
    if quality_or_compression is not None:
        if image_extension in available_extensions:
            imwrite_param = [available_extensions[image_extension], quality_or_compression]
        else:
            valid_extensions = str(list(available_extensions.keys()))
            raise RuntimeError(
                'Invalid image extension, must be one of (case-sensitive): %s' %
                valid_extensions)

    # Reset video manager and downscale factor.
    video_manager.release()
    video_manager.reset()
    video_manager.set_downscale_factor(downscale_factor)
    video_manager.start()

    # Setup flags and init progress bar if available.
    completed = True
    progress_bar = None
    if tqdm and show_progress:
        progress_bar = tqdm(
            total=len(scene_list) * num_images, unit='images')

    filename_template = Template(image_name_template)

    scene_num_format = '%0'
    scene_num_format += str(max(3, math.floor(math.log(len(scene_list), 10)) + 1)) + 'd'
    image_num_format = '%0'
    image_num_format += str(math.floor(math.log(num_images, 10)) + 2) + 'd'

    timecode_list = dict()

    for i in range(len(scene_list)):
        timecode_list[i] = []

    if num_images == 1:
        for i, (start_time, end_time) in enumerate(scene_list):
            duration = end_time - start_time
            timecode_list[i].append(start_time + int(duration.get_frames() / 2))
    else:
        middle_images = num_images - 2
        for i, (start_time, end_time) in enumerate(scene_list):
            timecode_list[i].append(start_time)

            if middle_images > 0:
                duration = (end_time.get_frames() - 1) - start_time.get_frames()
                duration_increment = None
                duration_increment = int(duration / (middle_images + 1))
                for j in range(middle_images):
                    timecode_list[i].append(start_time + ((j+1) * duration_increment))
            # End FrameTimecode is always the same frame as the next scene's start_time
            # (one frame past the end), so we need to subtract 1 here.
            timecode_list[i].append(end_time - 1)

    for i in timecode_list:
        for j, image_timecode in enumerate(timecode_list[i]):
            video_manager.seek(image_timecode)
            video_manager.grab()
            ret_val, frame_im = video_manager.retrieve()
            if ret_val:
                file_path = '%s.%s' % (filename_template.safe_substitute(
                    VIDEO_NAME=video_name,
                    SCENE_NUMBER=scene_num_format % (i + 1),
                    IMAGE_NUMBER=image_num_format % (j + 1)),
                                       image_extension)
                cv2.imwrite(
                    get_and_create_path(file_path, output_dir + '/' + 'scene' + str(i)),
                    frame_im, imwrite_param)
            else:
                completed = False
                break
            if progress_bar:
                progress_bar.update(1)

    return completed

