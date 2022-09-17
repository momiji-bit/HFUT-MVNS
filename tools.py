import cv2
import numpy as np
import torch
from utils.general import non_max_suppression, scale_coords, xyxy2xywh
from utils.augmentations import letterbox
from utils.general import check_img_size
from models.experimental import attempt_load
from deep_sort.utils.parser import get_config
from deep_sort.deep_sort import DeepSort
import time
import math
from pyquaternion import Quaternion
# ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
# 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant',
# 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard',
# 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle',
# 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
# 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv',
# 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator',
# 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']

# config
imgsz = 640  # 画面尺寸
weights = './yolov5n.pt'  # 预训练模型
device = 'cpu'  # 运算芯片:'cpu', 'cuda'
if torch.cuda.is_available():
    device = 'cuda'
half = False  # uint8 -> fp16/32
R = 800

# 初始化
stride, names = 64, [f'class{i}' for i in range(1000)]  # assign defaults

# 载入模型
model = attempt_load(weights, map_location=device)
stride = int(model.stride.max())  # model stride
names = model.module.names if hasattr(model, 'module') else model.names  # get class names
imgsz = check_img_size(imgsz, s=stride)  # check image size

# deepsort配置
cfg = get_config()
cfg.merge_from_file("deep_sort/configs/deep_sort.yaml")
deepsort = DeepSort(model_type='osnet_x0_25',
                    max_dist=cfg.DEEPSORT.MAX_DIST,
                    max_iou_distance=cfg.DEEPSORT.MAX_IOU_DISTANCE,
                    max_age=cfg.DEEPSORT.MAX_AGE,
                    n_init=cfg.DEEPSORT.N_INIT,
                    nn_budget=cfg.DEEPSORT.NN_BUDGET,
                    use_cuda=True)


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


class TrackPath():
    def __init__(self, lifetime=2):
        self.points = dict()
        self.lifetime = lifetime

    def updata(self, id, c, name, xyz, T):
        if id in self.points:
            self.points[id].append((xyz, T, c, name))
        else:
            self.points[id] = [(xyz, T, c, name)]

    def small(self):
        for i in self.points:
            # a b c 为最新三个点
            if len(self.points[i])>=3:
                a = self.points[i][-3]
                b = self.points[i][-2]
                c = self.points[i][-1]
                x_ab = math.sqrt((b[0][0]-a[0][0])**2 + (b[0][2]-a[0][2])**2)
                x_bc = math.sqrt((c[0][0]-b[0][0])**2 + (c[0][2]-b[0][2])**2)
                t_ab = b[1] - a[1]
                t_bc = c[1] - b[1]
                v_ab = (x_ab/t_ab)/1000  # m/s
                v_bc = (x_bc/t_bc)/1000
                accelerated = (v_bc-v_ab)/(t_bc-t_ab)
                # print(self.points[i][0][3],'  v1',v_ab,'  v2',v_bc,'  a',accelerated)
                if accelerated > 10 or accelerated < -10:  # 波尔特的最大加速度
                    del self.points[i][-1]
        for i in self.points:
            l = 0
            for j in self.points[i]:
                if time.time() - j[1] > self.lifetime:
                    l += 1
                else:
                    break
            del self.points[i][:l]

    def get_points(self):
        return self.points

    def predict(self, id):
        if id in self.points:
            # a b c 为最新三个点
            if len(self.points[id])>=3:
                a = self.points[id][-3]
                b = self.points[id][-2]
                c = self.points[id][-1]
                x_ab = math.sqrt((b[0][0]-a[0][0])**2 + (b[0][2]-a[0][2])**2)
                x_bc = math.sqrt((c[0][0]-b[0][0])**2 + (c[0][2]-b[0][2])**2)
                t_ab = b[1] - a[1]
                t_bc = c[1] - b[1]
                v_ab = (x_ab/t_ab)/1000  # m/s
                if c > b:
                    v_bc = -(x_bc/t_bc)/1000
                else:
                    v_bc = (x_bc / t_bc) / 1000
                accelerated = (v_bc-v_ab)/(t_bc-t_ab)  # m/s^2

                d_t = time.time()-c[1]
                p_l = v_bc * d_t + (accelerated * d_t ** 2)/2  # 预测移动距离
                print(f'{p_l:.3f}m {accelerated:.3f}m/s^2 {v_bc:.3f}m/s')


def track(RGB):
    out = []
    img_yolo = [letterbox(RGB, imgsz, stride=stride)[0]]
    # Stack
    img_yolo = np.stack(img_yolo, 0)
    # Convert
    img_yolo = img_yolo[..., ::-1].transpose((0, 3, 1, 2))  # BGR to RGB, BHWC to BCHW
    img_yolo = np.ascontiguousarray(img_yolo)  # 将内存不连续存储的数组转换为内存连续存储的数组，使得运行速度更快
    img_yolo = torch.from_numpy(img_yolo).to(device)  # 从numpy.ndarray创建一个张量
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
                if names[output[5]] in ['person']:
                    xyxy = output[0:4].tolist()  # bbox 坐标
                    id = output[4]  # 追踪编号
                    c = int(output[5])  # 类别
                    name = names[c]
                    t = [c, id, name, conf, xyxy]
                    out.append(t)
    else:
        deepsort.increment_ages()
    return out


def draw_birdView():
    frame = np.zeros((R*2, R*2, 3), np.uint8)
    cv2.circle(frame, (R, R), radius=2, color=(70, 70, 70), thickness=-1)
    cv2.circle(frame, (R, R), radius=100, color=(70, 70, 70), thickness=1)
    cv2.circle(frame, (R, R), radius=300, color=(70, 70, 70), thickness=1)
    cv2.circle(frame, (R, R), radius=500, color=(70, 70, 70), thickness=1)
    cv2.circle(frame, (R, R), radius=800, color=(70, 70, 70), thickness=1)
    return frame

