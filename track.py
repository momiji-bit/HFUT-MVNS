import depthai as dai
import cv2
import numpy as np
import torch
import math
from utils.general import non_max_suppression, scale_coords, xyxy2xywh
from utils.plots import Annotator, colors
from utils.augmentations import letterbox
from parameter import imgsz, stride,half, \
    model, line_thickness, names
from deep_sort.utils.parser import get_config
from deep_sort.deep_sort import DeepSort



# deepsort参数配置
cfg = get_config()
cfg.merge_from_file("deep_sort/configs/deep_sort.yaml")
deepsort = DeepSort(model_type='osnet_x0_25',
                    max_dist=cfg.DEEPSORT.MAX_DIST,
                    max_iou_distance=cfg.DEEPSORT.MAX_IOU_DISTANCE,
                    max_age=cfg.DEEPSORT.MAX_AGE,
                    n_init=cfg.DEEPSORT.N_INIT,
                    nn_budget=cfg.DEEPSORT.NN_BUDGET,
                    use_cuda=False)
class Colors:
    # Ultralytics color palette https://ultralytics.com/
    def __init__(self):
        # hex = matplotlib.colors.TABLEAU_COLORS.values()
        hex = ('FF3838', 'FF9D97', 'FF701F', 'FFB21D', 'CFD231', '48F90A', '92CC17', '3DDB86', '1A9334', '00D4BB',
               '2C99A8', '00C2FF', '344593', '6473FF', '0018EC', '8438FF', '520085', 'CB38FF', 'FF95C8', 'FF37C7')
        self.palette = [self.hex2rgb('#' + c) for c in hex]
        self.n = len(self.palette)

    def __call__(self, i, bgr=False):
        c = self.palette[int(i) % self.n]
        return (c[2], c[1], c[0]) if bgr else c

    @staticmethod
    def hex2rgb(h):  # rgb order (PIL)
        return tuple(int(h[1 + i:1 + i + 2], 16) for i in (0, 2, 4))


def track(RGB):
    out = []
    img_yolo = [letterbox(RGB, imgsz, stride=stride)[0]]
    # Stack
    img_yolo = np.stack(img_yolo, 0)
    # Convert
    img_yolo = img_yolo[..., ::-1].transpose((0, 3, 1, 2))  # BGR to RGB, BHWC to BCHW
    img_yolo = np.ascontiguousarray(img_yolo)  # 将内存不连续存储的数组转换为内存连续存储的数组，使得运行速度更快
    img_yolo = torch.from_numpy(img_yolo).to('cpu')  # 从numpy.ndarray创建一个张量
    img_yolo = img_yolo.half() if half else img_yolo.float()  # uint8 to fp16/32
    img_yolo = img_yolo / 255.0  # 0 - 255 to 0.0 - 1.0
    pred = model(img_yolo)[0]  # 预测  640 416
    det = non_max_suppression(prediction=pred, conf_thres=0.3, iou_thres=0.45, max_det=20)[0]  # NMS非极大值抑制

    if len(det):
        # Rescale boxes from img_size to im0 size
        det[:, :4] = scale_coords(img_yolo.shape[2:], det[:, :4], RGB.shape).round()
        xywhs = xyxy2xywh(det[:, 0:4])
        confs = det[:, 4]
        clss = det[:, 5]
        outputs = deepsort.update(xywhs.cpu(), confs.cpu(), clss.cpu(), RGB)
        if len(outputs) > 0:
            for j, (output, conf) in enumerate(zip(outputs, confs)):
                xyxy = output[0:4].tolist()  # bbox 坐标
                id = output[4]  # 追踪编号
                c = int(output[5])  # 类别
                name = names[c]
                t = [c, id, name, conf, xyxy]
                out.append(t)
    else:
        deepsort.increment_ages()
    return out





