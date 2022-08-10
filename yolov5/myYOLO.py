import numpy as np
from yolov5.utils.augmentations import letterbox
import os
import sys
from pathlib import Path
from yolov5.models.common import DetectMultiBackend
from yolov5.utils.general import (check_img_size, increment_path, non_max_suppression, scale_coords)
from yolov5.utils.plots import Annotator, colors
from yolov5.utils.torch_utils import select_device
import cv2
import torch

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative



weights=ROOT / 'yolov5n.pt'  # model.pt path(s)
# source=ROOT / 'data/images'  # file/dir/URL/glob, 0 for webcam
data=ROOT / 'data/coco128.yaml'  # dataset.yaml path
imgsz=(640, 640)  # inference size (height, width)
conf_thres=0.25  # confidence threshold
iou_thres=0.45  # NMS IOU threshold
max_det=1000  # maximum detections per image
device=''  # cuda device, i.e. 0 or 0,1,2,3 or cpu
# view_img=False  # show results
save_txt=False  # save results to *.txt
save_conf=False  # save confidences in --save-txt labels
# save_crop=False  # save cropped prediction boxes
nosave=False  # do not save images/videos
classes=None  # filter by class: --class 0, or --class 0 2 3
agnostic_nms=False  # class-agnostic NMS
augment=False  # augmented inference
visualize=False  # visualize features
update=False  # update all models
project=ROOT / 'runs/detect'  # save results to project/name
name='exp'  # save results to project/name
exist_ok=False  # existing project/name ok, do not increment
line_thickness=3  # bounding box thickness (pixels)
hide_labels=False  # hide labels
hide_conf=False  # hide confidences
half=False  # use FP16 half-precision inference
dnn=False  # use OpenCV DNN for ONNX inference
img_size=640
stride=32
auto=True

# Directories
save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
(save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir
# Load model
device = select_device(device)
model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data)
stride, names, pt, jit, onnx, engine = model.stride, model.names, model.pt, model.jit, model.onnx, model.engine
imgsz = check_img_size(imgsz, s=stride)  # check image size

# Half
half &= (pt or jit or onnx or engine) and device.type != 'cpu'  # FP16 supported on limited backends with CUDA
if pt or jit:
    model.model.half() if half else model.model.float()


def pred(frame):

    # Letterbox
    s = np.stack([letterbox(frame, img_size, stride=stride, auto=auto)[0].shape])
    rect = np.unique(s, axis=0).shape[0] == 1  # rect inference if all shapes equal
    img = [letterbox(frame, img_size, stride=stride, auto=rect and auto)[0]]

    # Stack
    img = np.stack(img, 0)

    # Convert
    img = img[..., ::-1].transpose((0, 3, 1, 2))  # BGR to RGB, BHWC to BCHW
    img = np.ascontiguousarray(img)
    im = img
    im = torch.from_numpy(im).to(device)
    im = im.half() if half else im.float()  # uint8 to fp16/32
    im /= 255  # 0 - 255 to 0.0 - 1.0
    if len(im.shape) == 3:
        im = im[None]  # expand for batch dim

    # Inference
    # visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
    pred = model(im, augment=augment, visualize=visualize)


    # NMS
    pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)

    # Second-stage classifier (optional)
    # pred = utils.general.apply_classifier(pred, classifier_model, im, im0s)

    # Process predictions
    # for i, det in enumerate(pred):  # per image
    #     im0 = frame
    #     gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh\
    #     annotator = Annotator(im0, line_width=line_thickness, example=str(names))
    #     if len(det):
    #         # Rescale boxes from img_size to im0 size
    #         det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()
    #         # Print results
    #         for c in det[:, -1].unique():
    #             n = (det[:, -1] == c).sum()  # detections per class
    #             # s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string
    #         # Write results
    #         for *xyxy, conf, cls in reversed(det):
    #             c = int(cls)  # integer class
    #             label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
    #             annotator.box_label(xyxy, label, color=colors(c, True))
    #     # Stream results
    #     im0 = annotator.result()
    #     cv2.imshow('out', im0)
    #     cv2.waitKey(1)  # 1 millisecond
    return pred, im, names
