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
import time
import matplotlib.pyplot as plt
import matplotlib


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

def getFrame(queue):
    """获取队列中最后一帧画面"""
    # Get frame from queue
    frame = queue.get()
    # Convert frame to OpenCV format and return
    return frame.getCvFrame()


def getCamera(pipline, isMono, isLeft=None):
    """获取相机对象"""
    # Configure mono camera
    if isMono:
        mono = pipline.createMonoCamera()
        # Set camera resolution
        mono.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
        if isLeft:
            # Get left camera
            mono.setBoardSocket(dai.CameraBoardSocket.LEFT)
        else:
            mono.setBoardSocket(dai.CameraBoardSocket.RIGHT)
        return mono
    else:
        colorCam = pipline.createColorCamera()
        colorCam.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
        colorCam.setBoardSocket(dai.CameraBoardSocket.RGB)
        return colorCam


def getStereoPair(pipeline, monoLeft, monoRight):
    """获取自带立体深度节点 对象"""
    # Configure stereo pair for depth estimation
    stereo = pipeline.createStereoDepth()
    # Checks occluded pixels and marks them as invalid
    # 用于计算哪些像素被遮挡了，将这些像素设为0
    stereo.setLeftRightCheck(True)
    stereo.setSubpixel(False)  # 适合远距离物体
    stereo.setExtendedDisparity(False)  # 适合近距离物体

    # Configure left and right cameras to work as a stereo pair
    monoLeft.out.link(stereo.left)
    monoRight.out.link(stereo.right)
    return stereo


def getX(z, x):
    fov = 68.7938
    alpha = fov / 2
    x_c = x - 320
    w = math.tan(math.radians(alpha)) * z
    return (w * x_c) / 640


def make_bird_frame():
    max_z = 16  # 距离
    min_z = 0
    fov = 68.7938
    min_distance = 0.827
    frame = np.zeros((1600, 1000, 3), np.uint8)
    # min_y = int((1 - (min_distance - min_z) / (max_z - min_z)) * frame.shape[0])  # frame.shape[0] 竖向长度
    # cv2.rectangle(frame, (0, min_y), (frame.shape[1], frame.shape[0]), (70, 70, 70), -1)

    alpha = (180 - fov) / 2
    center = int(frame.shape[1] / 2)
    max_p = frame.shape[0] - int(math.tan(math.radians(alpha)) * center)
    fov_cnt = np.array([
        (0, frame.shape[0]),
        (frame.shape[1], frame.shape[0]),
        (frame.shape[1], max_p),
        (center, frame.shape[0]),
        (0, max_p),
        (0, frame.shape[0]),
    ])
    cv2.fillPoly(frame, [fov_cnt], color=(70, 70, 70))
    return frame


def getObjectDepth(depth, xyxy, color):  # 640 400
    slice = depth[xyxy[1]:xyxy[3], xyxy[0]:xyxy[2]]
    slice = slice.flatten()
    slice.sort()
    l = len(slice)
    slice = slice[int(l / 5):int(l / 4)]
    return np.mean(slice)

def draw(data):
    plt.figure(dpi=500)  # 设置画布的大小和dpi，为了使图片更加清晰
    ax = plt.axes()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.xlabel('time (ms)')  # x轴标签
    plt.ylabel('distance (mm)')  # y轴标签

    for i in data:
        RGB = np.random.randint(0,255,3).tolist()
        color = ''
        for c in RGB:
            t = hex(c)[2:]
            if len(t)==2:
                color+=t
            else:
                color+='0'+t
        color = '#'+color
        mess = data[i]
        x_time = []
        y_distance = []
        for t in mess:
            x_time.append(t[0])
            y_distance.append(t[1])
        plt.plot(x_time, y_distance, linewidth=1, linestyle="solid", label=i, color=color)

    plt.legend()
    plt.title('tracker')
    # plt.show()

    plt.savefig('./fig.png')  # 保存图片


def draw_c(data):
    plt.figure(dpi=500)  # 设置画布的大小和dpi，为了使图片更加清晰
    ax = plt.axes()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.xlabel('LR')  # x轴标签
    plt.ylabel('distance')  # y轴标签

    for i in data:
        RGB = np.random.randint(0,255,3).tolist()
        color = ''
        for c in RGB:
            t = hex(c)[2:]
            if len(t)==2:
                color+=t
            else:
                color+='0'+t
        color = '#'+color
        mess = data[i]
        x = []
        y = []
        for t in mess:
            x.append(t[2])
            y.append(t[1])
        plt.plot(x, y, linewidth=1, linestyle="solid", label=i, color=color)

    plt.legend()
    plt.title('tracker')
    # plt.show()

    plt.savefig('./fig_XY.png')  # 保存图片



def yolo_deepsort_sgbm():
    global colorImg, depth, out_img, depthDemo, place, t_start, DATA_DRAW
    img_yolo = [letterbox(colorImg, imgsz, stride=stride)[0]]
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

    s = 'cam'
    s += '[%gx%g]: ' % img_yolo.shape[2:]  # print string
    # gn = torch.tensor(img0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
    annotator = Annotator(out_img, line_width=line_thickness, example=str(names))  # 创建bbox绘制工具

    if len(pred):

        # Rescale boxes from img_size to im0 size
        det[:, :4] = scale_coords(img_yolo.shape[2:], det[:, :4], out_img.shape).round()
        xywhs = xyxy2xywh(det[:, 0:4])
        confs = det[:, 4]
        clss = det[:, 5]
        outputs = deepsort.update(xywhs.cpu(), confs.cpu(), clss.cpu(), colorImg)
        if len(outputs) > 0:
            for j, (output, conf) in enumerate(zip(outputs, confs)):

                xyxy = output[0:4]
                id = output[4]
                cls = output[5]
                c = int(cls)  # integer class
                # 计算目标距离
                dist = getObjectDepth(depth, (xyxy[0], xyxy[1], xyxy[2], xyxy[3]), colors(c, True))
                # 创建Label（预测框，物品名称，置信度，距离信息）
                label = f'{names[c]}{id} {conf:.0%} {dist / 1000:.4}m'
                s += f'[{label}]\t'
                annotator.box_label(xyxy, label, color=colors(c, True))
                cv2.rectangle(depthDemo, (xyxy[0], xyxy[1]), (xyxy[2], xyxy[3]), (255, 255, 255), 1)

                if f'{id}-{names[c]}' in DATA_DRAW:
                    DATA_DRAW[f'{id}-{names[c]}'].append([int(time.time()*100)-t_start, int(dist), getX(dist, (xyxy[0]+xyxy[2])/2)])
                else:
                    DATA_DRAW[f'{id}-{names[c]}'] = [[int(time.time() * 100) - t_start, int(dist), getX(dist, (xyxy[0]+xyxy[2])/2)]]

                cp = (int(place.shape[1]/2 + getX(dist, (xyxy[0]+xyxy[2])/2)/10), place.shape[0] - int(dist/10))
                r = max(1, int((getX(dist, xyxy[2])-getX(dist, xyxy[0]))/20))
                cv2.circle(place, cp, radius=r,
                           color=colors(c, True), thickness=-1)
                cv2.putText(place, f'{id} {names[c]}', (cp[0]+r,cp[1]-r),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color=(255, 255, 255), thickness=1)
    else:
        deepsort.increment_ages()
    # draw(DATA_DRAW)



if __name__ == '__main__':
    # 初始化管道
    pipeline = dai.Pipeline()

    colorImg = None
    depth = None
    out_img = None
    depthDemo = None
    place = None
    t_start = int(time.time()*100)  # ms*10
    DATA_DRAW = dict()
    DATA_DRAW_C = dict()

    # 获取左右黑白相机
    monoLeft = getCamera(pipeline, isMono=True, isLeft=True)
    monoRight = getCamera(pipeline, isMono=True, isLeft=False)
    colorCam = getCamera(pipeline, isMono=False)

    # 将左右相机数据用于Stereo pair对象的输入
    stereo = getStereoPair(pipeline, monoLeft, monoRight)

    # 为disparity, rectifiedLeft, rectifiedRight 设置XlinkOut
    xoutDepth = pipeline.createXLinkOut()
    xoutDepth.setStreamName('depth')

    xoutrectifiedLeft = pipeline.createXLinkOut()
    xoutrectifiedLeft.setStreamName('rectifiedLeft')
    xoutrectifiedRight = pipeline.createXLinkOut()
    xoutrectifiedRight.setStreamName('rectifiedRight')
    xoutColorImg = pipeline.createXLinkOut()
    xoutColorImg.setStreamName("colorImg")

    # stereo.disparity.link(xoutDisp.input)
    stereo.depth.link(xoutDepth.input)
    stereo.rectifiedLeft.link(xoutrectifiedLeft.input)
    stereo.rectifiedRight.link(xoutrectifiedRight.input)
    colorCam.preview.link(xoutColorImg.input)
    colorCam.setPreviewSize(640, 400)




    # 管道初始化完成，现在可以连接设备
    with dai.Device(pipeline) as device:
        # 输出队列将会用于获取RGB图像
        depthQueue = device.getOutputQueue(name='depth', maxSize=1, blocking=False)
        rectifiedLeftQueue = device.getOutputQueue(name='rectifiedLeft', maxSize=1, blocking=False)
        rectifiedRightQueue = device.getOutputQueue(name='rectifiedRight', maxSize=1, blocking=False)
        colorImgQueue = device.getOutputQueue(name="colorImg", maxSize=1, blocking=False)


        while True:
            t0 = time.time()
            place = make_bird_frame()

            depth = getFrame(depthQueue)  # 640 400
            depth = depth[60:340, 62:510]
            depth = cv2.resize(depth, (640, 400))
            depthDemo = cv2.normalize(depth, None, 255, 0, cv2.NORM_INF, cv2.CV_8UC1)
            depthDemo = cv2.equalizeHist(depthDemo)
            depthDemo = cv2.applyColorMap(255 - depthDemo, cv2.COLORMAP_JET)
            colorImg = getFrame(colorImgQueue)  # 640 400
            out_img = colorImg.copy()

            # yolo deepsort
            yolo_deepsort_sgbm()

            # myBisenet()

            t = time.time()
            cv2.putText(out_img, "FPS: {:.2f}".format(1 / (t - t0)), (2, colorImg.shape[0] - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 1)
            out_img = cv2.resize(out_img, (1280, 800))
            depthDemo = cv2.resize(depthDemo, (1280, 800))
            place = cv2.resize(place, (500, 800))
            # out = np.hstack((out_img, place, depthDemo))
            out = np.hstack((out_img, depthDemo))
            # out = np.uint8(out_img / 2 + depthDemo / 2)
            # cv2.imshow('RGB', colorImg)
            # cv2.imshow('depth', depthDemo)  # 640 400
            # cv2.imshow('Demo', out_img)
            cv2.imshow('palce', place)
            cv2.imshow('out', out)
            if cv2.waitKey(1) == ord('q'):
                draw(DATA_DRAW)
                draw_c(DATA_DRAW)
                break





