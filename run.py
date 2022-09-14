# 矫正后的彩色画面与深度图，边缘重合

import cv2
import depthai as dai
import numpy as np
import time
from track import track, Colors
import matplotlib.pyplot as plt
import math


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
        v_bc = 0
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

        return v_bc


def point_to_ConfigData(topLeft_px, bottomRight_px, imgSize_hw=[640, 360], lowerThreshold=0, upperThreshold=20000):
    topLeft = dai.Point2f(topLeft_px[0] / imgSize_hw[0], topLeft_px[1] / imgSize_hw[1])
    bottomRight = dai.Point2f(bottomRight_px[0] / imgSize_hw[0], bottomRight_px[1] / imgSize_hw[1])
    config = dai.SpatialLocationCalculatorConfigData()
    config.depthThresholds.lowerThreshold = lowerThreshold
    config.depthThresholds.upperThreshold = upperThreshold
    config.roi = dai.Rect(topLeft, bottomRight)
    return config

def getX(z, x):
    fov = 68.7938
    alpha = fov / 2
    x_c = x - 320
    w = math.tan(math.radians(alpha)) * z
    return (w * x_c) / 640

def getObjectDepth(depth, xyxy):  # 640 400
    slice = depth[xyxy[1]:xyxy[3], xyxy[0]:xyxy[2]]
    slice = slice.flatten()
    slice.sort()
    l = len(slice)
    slice = slice[int(l / 5):int(l / 4)]
    return np.mean(slice)

def draw_birdView():
    fov = 68.7938
    frame = np.zeros((1600, 1000, 3), np.uint8)
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

    fontType = cv2.FONT_HERSHEY_TRIPLEX
    color = (255,255,255)
    cv2.putText(frame, '-0M', (0, 1600), fontType, 0.8, color, 1)
    cv2.putText(frame, '-1M', (0, 1500), fontType, 0.8, color, 1)
    cv2.putText(frame, '-2M', (0, 1400), fontType, 0.8, color, 1)
    cv2.putText(frame, '-3M', (0, 1300), fontType, 0.8, color, 1)
    cv2.putText(frame, '-5M', (0, 1100), fontType, 0.8, color, 1)
    cv2.putText(frame, '-9M', (0, 700), fontType, 0.8, color, 1)
    cv2.putText(frame, '-15M', (0, 100), fontType, 0.8, color, 1)

    return frame


def getFrame(queue):
    """获取队列中最后一帧画面"""
    # Get frame from queue
    frame = queue.get()
    # Convert frame to OpenCV format and return
    return frame.getCvFrame()

colors = Colors()
path = TrackPath(lifetime=4)


def go(color, depth):
    t0 = time.time()
    depthDemo = cv2.normalize(depth, None, 255, 0, cv2.NORM_INF, cv2.CV_8UC1)
    depthDemo = cv2.equalizeHist(depthDemo)
    depthDemo = cv2.applyColorMap(255 - depthDemo, cv2.COLORMAP_HOT)

    out = track(color)  # t = [c, id, name, conf, xyxy]
    background = draw_birdView()

    for i in out:

        col = colors(i[0], True)
        cv2.rectangle(color, tuple(i[4][:2]), tuple(i[4][2:]), col, 1)
        lable = f'{i[2]}-{i[1]} {i[3]:.0%}'
        cv2.putText(color, lable, (i[4][0] + 2, i[4][1] + 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2)


        Z = getObjectDepth(depth, i[4])
        X = getX(Z, (i[4][0]+i[4][2])/2)
        Y = 0
        xmin = i[4][0]
        ymin = i[4][1]
        cv2.rectangle(depthDemo, tuple(i[4][:2]), tuple(i[4][2:]), col, 1)
        cv2.putText(depthDemo, f"X: {X:.0f} mm",
                    (xmin + 10, ymin + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 0, 255))
        cv2.putText(depthDemo, f"Y: {Y:.0f} mm",
                    (xmin + 10, ymin + 35), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 0, 255))
        cv2.putText(depthDemo, f"Z: {Z:.0f} mm",
                    (xmin + 10, ymin + 50), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 0, 255))
        #
        path.updata(id=i[1], c=i[0], name=i[2], xyz=(X, Y, Z), T=time.time())
        path.small()
        v = path.predict(i[1])

        cv2.putText(color, f"v:{v:.2f} m/s",
                    (xmin + 2, ymin + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255))

    points = path.get_points()
    path.small()
    for t in points:
        P = points[t]
        onePath = []
        c = None
        for point in P:
            c = point[2]
            cp = (int((background.shape[1] / 2 + point[0][0] / 10)), int(background.shape[0] - point[0][2] / 10))
            cv2.circle(background, cp, radius=3, color=colors(c, True), thickness=-1)
            onePath.append(cp)
        if len(onePath) > 1:
            cv2.polylines(background, [np.array(onePath)], isClosed=False, color=colors(c, True), thickness=1)
    cv2.line(background, (0, 1500), (1000, 1500), (100,100,100), 1)

    cv2.putText(color, "FPS: {:.1f}".format(1 / (time.time() - t0)), (2, color.shape[0] - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)
    # cv2.line(depthDemo, (200, 0), (200, 360), (255, 0, 255), 1)
    # cv2.line(color, (320, 0), (320, 360), (255, 0, 255), 1)
    out = np.vstack((color, depthDemo))
    background = cv2.resize(background,(469,720))
    out = np.hstack((background, out))
    # cv2.imshow('depth', depthDemo)
    # cv2.imshow('color', color)
    # cv2.imshow('background', background)
    cv2.imshow('out', out)
