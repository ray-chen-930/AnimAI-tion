import torch
import numpy as np

import utils.datasets
import utils.general
import utils.torch_utils
import models.experimental
import utils.google_utils
import models

def load_model(weights):
    utils.general.set_logging()
    # device = select_device(opt.device)
    device = utils.torch_utils.select_device('cpu')
    half = device.type != 'cpu'
    return models.experimental.attempt_load(weights, map_location=device), device, half


def extract_bbox_from_frame(frame, model, device, half, imgsz=640, to_yield_for_list=True):
    imgsz = utils.general.check_img_size(imgsz, s=model.stride.max())  # check img_size
    if half:
        model.half()  # to FP16

    img = utils.datasets.letterbox(frame, new_shape=imgsz)[0]
    img = img[:, :, ::-1].transpose(2, 0, 1)
    img = np.ascontiguousarray(img)

    # Run inference
    img = torch.from_numpy(img).to(device)
    img = img.half() if half else img.float()  # uint8 to fp16/32
    img /= 255.0  # 0 - 255 to 0.0 - 1.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)

    # Inference
    pred = model(img, augment=True)[0]

    # Apply NMS
    pred = utils.general.non_max_suppression(pred, .25, .45, classes=0, agnostic=True)

    # Process detections
    im0 = frame
    for det in pred:  # detections per image
        gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
        if det is not None and len(det):
            # Rescale boxes from img_size to im0 size
            det[:, :4] = utils.general.scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

            if to_yield_for_list:
                for (x, y, xm, ym, conf, cls) in det:
                    # skip if bbox score is less than .75
                    if conf < .75:
                        continue
                    yield [float(x), float(y), float(xm), float(ym), float(conf)]
