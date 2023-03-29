import os

# limit the number of cpus used by high performance libraries
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import sys
import numpy as np
from pathlib import Path
import torch

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # yolov5 strongsort root directory
WEIGHTS = ROOT / 'weights'

if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
if str(ROOT / 'yolov5') not in sys.path:
    sys.path.append(str(ROOT / 'yolov5'))  # add yolov5 ROOT to PATH
if str(ROOT / 'trackers' / 'strongsort') not in sys.path:
    sys.path.append(str(ROOT / 'trackers' / 'strongsort'))  # add strong_sort ROOT to PATH

ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from yolov8.ultralytics.nn.autobackend import AutoBackend
from yolov8.ultralytics.yolo.data.augment import LetterBox
from yolov8.ultralytics.yolo.utils.torch_utils import select_device
from yolov8.ultralytics.yolo.utils.ops import non_max_suppression
from yolov8.ultralytics.nn.autobackend import AutoBackend
from yolov8.ultralytics.yolo.data.dataloaders.stream_loaders import LoadImages, LoadStreams
from yolov8.ultralytics.yolo.data.utils import IMG_FORMATS, VID_FORMATS
from yolov8.ultralytics.yolo.utils import DEFAULT_CFG, LOGGER, SETTINGS, callbacks, colorstr, ops
from yolov8.ultralytics.yolo.utils.checks import check_file, check_imgsz, check_imshow, print_args, check_requirements
from yolov8.ultralytics.yolo.utils.files import increment_path
from yolov8.ultralytics.yolo.utils.torch_utils import select_device, strip_optimizer
from yolov8.ultralytics.yolo.utils.ops import Profile, non_max_suppression, scale_boxes, process_mask, process_mask_native
from yolov8.ultralytics.yolo.utils.plotting import Annotator, colors, save_one_box

from trackers.multi_tracker_zoo import create_tracker

# model setting
yolo_weights = WEIGHTS / 'yolov8s-seg.pt'  # model path or triton URL
reid_weights = WEIGHTS / 'osnet_x0_25_msmt17.pt'
tracking_method = 'bytetrack'
tracking_config = ROOT / 'trackers' / tracking_method / 'configs' / (tracking_method + '.yaml')
imgsz = [640, 640]  # inference size (height, width)
device = ''  # cuda device, i.e. 0 or 0,1,2,3 or cpu
dnn = False  # use OpenCV DNN for ONNX inference
data = ROOT / 'data/coco128.yaml'  # dataset.yaml path
half = False  # use FP16 half-precision inference
visualize = False

device = select_device(device)
is_seg = '-seg' in str(yolo_weights)
model = AutoBackend(yolo_weights, device=device, dnn=dnn, fp16=half)
stride, names, pt = model.stride, model.names, model.pt
# imgsz = check_imgsz(imgsz, stride=stride)  # check image size
bs = 1  # batch_size
vid_path, vid_writer, txt_path = [None] * bs, [None] * bs, [None] * bs
model.warmup(imgsz=(1 if pt or model.triton else bs, 3, *imgsz))  # warmup

# Create as many strong sort instances as there are video sources
tracker_list = []
for i in range(bs):
    tracker = create_tracker(tracking_method, tracking_config, reid_weights, device, half)
    tracker_list.append(tracker, )
    if hasattr(tracker_list[i], 'model'):
        if hasattr(tracker_list[i].model, 'warmup'):
            tracker_list[i].model.warmup()
outputs = [None] * bs


def run_img(img,
            img_size=640,
            stride=32,
            auto=True,
            # inference setting
            augment=False,  # augmented inference
            conf_thres=0.25,  # confidence threshold
            iou_thres=0.45,  # NMS IOU threshold
            max_det=1000,  # maximum detections per image
            classes=None,  # filter by class: --class 0, or --class 0 2 3
            agnostic_nms=False,  # class-agnostic NMS
            line_thickness=3,  # bounding box thickness (pixels)
            hide_labels=False,  # hide labels
            hide_conf=False,  # hide confidences
            ):
    im0 = [None]
    im0[0] = img
    im = np.stack([LetterBox(imgsz, auto, stride=stride)(image=x) for x in im0])
    im = im[..., ::-1].transpose((0, 3, 1, 2))  # BGR to RGB, BHWC to BCHW
    im = np.ascontiguousarray(im)  # contiguous

    im = torch.from_numpy(im).to(device)
    im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
    im /= 255.0  # 0 - 255 to 0.0 - 1.0
    if len(im.shape) == 3:
        im = im[None]  # expand for batch dim

    # Inference
    preds = model(im, augment=augment, visualize=visualize)

    # Apply NMS
    if is_seg:
        masks = []
        p = non_max_suppression(preds[0], conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det, nm=32)
        proto = preds[1][-1]
    else:
        p = non_max_suppression(preds, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)

    return p, img, im0, im, masks, proto
