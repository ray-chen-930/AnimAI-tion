import numpy as np
import matplotlib.pyplot as plt
import cv2

bbox_path = 'data_train_yolo/labels/train/images.txt'
im_path = 'data_train_yolo/images/train/images.jpeg'

# bbox_path = 'data_train_yolo/labels/train/im_test.txt'
# im_path = 'data_train_yolo/images/train/im_test.jpg'

def get_bbox(path):
    # bbox coordinates as (x of center, y of center, width of box, height of box) all in percentage of im
    bbox = []
    with open(path, 'r') as f:
        for line in f:
            line = line.rstrip()
            l = line.split(' ')
            bbox.append([0] + l[1:])
    return [list(map(float, bb)) for bb in bbox]


def get_im(path):
    im = cv2.imread(path)
    im = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)
    return im

def draw_bbox(im, bbox_list, start_idx=1):
    im_h = im.shape[0]
    im_w = im.shape[1]
    for b in bbox_list:
        b_w = int(b[2 + start_idx] * im_w)
        b_h = int(b[3 + start_idx] * im_h)
        center = (int(b[0 + start_idx] * im_w), int(b[1 + start_idx] * im_h))
        top = int(center[1] - b_h/2)
        bot = int(center[1] + b_h / 2)
        left = int(center[0] - b_w / 2)
        right = int(center[0] + b_w / 2)

        new_im = cv2.rectangle(im, (left, top),
                               (right, bot),
                               color=(0, 255, 0), thickness=2)
    return new_im

if __name__ == "__main__":
    bbox_list = get_bbox(bbox_path)
    im = get_im(im_path)
    im_boxed = draw_bbox(im, bbox_list)


print(9)