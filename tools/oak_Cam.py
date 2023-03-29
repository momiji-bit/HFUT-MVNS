# 仅展示OAK相机的使用教程 显示相机的彩色和深度图 并已进行了数据同步（彩色和深度同步）
import math
import time

import cv2
import depthai as dai
import matplotlib
import numpy as np

matplotlib.use('TKAgg')

tan = math.tan
radians = math.radians

hfov = 68.7938003540039
vfov = 42.12409823672219


class OakCam:
    class HostSync:  # RGB D 帧同步
        def __init__(self):
            self.arrays = {}

        def add_msg(self, name, msg):
            if not name in self.arrays:
                self.arrays[name] = []
            # Add msg to array
            self.arrays[name].append({'msg': msg, 'seq': msg.getSequenceNum()})
            # print(f'\rname: {name}, seq: {msg.getSequenceNum()}', end='')

            synced = {}
            for name, arr in self.arrays.items():
                for i, obj in enumerate(arr):
                    if msg.getSequenceNum() == obj['seq']:
                        synced[name] = obj['msg']
                        break
            # If there are 3 (all) synced msgs, remove all old msgs
            # and return synced msgs
            if len(synced) == 2:  # color, depth, nn
                # Remove old msgs
                for name, arr in self.arrays.items():
                    for i, obj in enumerate(arr):
                        if obj['seq'] < msg.getSequenceNum():
                            arr.remove(obj)
                        else:
                            break
                return synced
            return False

    def __init__(self):
        # Create pipeline
        self.pipeline = dai.Pipeline()

        # This might improve reducing the latency on some systems
        self.pipeline.setXLinkChunkSize(0)

        # Define source and output
        camRgb = self.pipeline.create(dai.node.ColorCamera)
        camRgb.setFps(30)
        camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_720_P)
        camRgb.setIspScale(2, 3)

        stereo = self.pipeline.create(dai.node.StereoDepth)
        # stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)
        stereo.initialConfig.setMedianFilter(dai.MedianFilter.KERNEL_7x7)
        stereo.setLeftRightCheck(True)
        stereo.setDepthAlign(dai.CameraBoardSocket.RGB)
        stereo.setExtendedDisparity(False)
        stereo.setSubpixel(False)

        camMNR = self.pipeline.create(dai.node.MonoCamera)
        camMNR.setFps(30)
        camMNR.setResolution(dai.MonoCameraProperties.SensorResolution.THE_720_P)
        camMNR.setBoardSocket(dai.CameraBoardSocket.RIGHT)
        camMNR.out.link(stereo.right)

        camMNL = self.pipeline.create(dai.node.MonoCamera)
        camMNL.setFps(30)
        camMNL.setResolution(dai.MonoCameraProperties.SensorResolution.THE_720_P)
        camMNL.setBoardSocket(dai.CameraBoardSocket.LEFT)
        camMNL.out.link(stereo.left)

        imu = self.pipeline.create(dai.node.IMU)
        imu.enableIMUSensor([dai.IMUSensor.ACCELEROMETER_RAW, dai.IMUSensor.GYROSCOPE_RAW], 100)
        imu.setBatchReportThreshold(1)
        imu.setMaxBatchReports(10)

        xoutRGB = self.pipeline.create(dai.node.XLinkOut)
        xoutRGB.setStreamName("camRgb")
        camRgb.isp.link(xoutRGB.input)

        xoutD = self.pipeline.create(dai.node.XLinkOut)
        xoutD.setStreamName("Depth")
        stereo.depth.link(xoutD.input)

        xoutIMU = self.pipeline.create(dai.node.XLinkOut)
        xoutIMU.setStreamName("IMU")
        imu.out.link(xoutIMU.input)

    def get_rgbCam_w(self):
        return self.w

    def get_rgbCam_h(self):
        return self.h

    def get_rgbCam_fps(self):
        return self.fps

    def getObjectDepth(self, depth, xyxy):  # 640 360
        slice = depth[xyxy[1]:xyxy[3], xyxy[0]:xyxy[2]]
        slice = slice.flatten()
        slice.sort()
        l = len(slice)
        slice = slice[int(l / 3):int(l * 2 / 3)]

        num = [0]*10001
        for i in slice:
            num[i] += 1

        return np.mean(slice), num  # 前+后-

    def getObjectX(self, z, x):
        x_c = x - 320
        w = tan(radians(hfov / 2)) * z
        return -(w * x_c) / 320  # 左+右-

    def getObjectY(self, z, y):
        y_c = y - 180
        w = tan(radians(vfov / 2)) * z
        return -(w * y_c) / 180  # 上+下-

    def get_msg(self):
        with dai.Device(self.pipeline) as device:
            device.setIrLaserDotProjectorBrightness(0)  # in mA, 0..1200
            device.setIrFloodLightBrightness(0)  # in mA, 0..1500

            q_camRgb = device.getOutputQueue(name="camRgb", maxSize=1, blocking=False)
            q_Depth = device.getOutputQueue(name="Depth", maxSize=1, blocking=False)
            q_IMU = device.getOutputQueue(name="IMU", maxSize=1, blocking=False)
            while True:
                imgRgb = q_camRgb.get()
                Depth = q_Depth.get()
                IMU = q_IMU.get()
                yield imgRgb.getCvFrame(), Depth.getFrame(), IMU, \
                    (dai.Clock.now() - imgRgb.getTimestamp()).total_seconds(), \
                    (dai.Clock.now() - Depth.getTimestamp()).total_seconds()


if __name__ == '__main__':
    oakCam = OakCam()
    msg = oakCam.get_msg()
    while True:
        t0 = time.time()
        RGB, D, IMU, latencyRGB, latencyD = next(msg)
        print(1/latencyRGB, 1/latencyD)
        cv2.imshow('RGB', RGB)
        cv2.imshow('D', D)
        if cv2.waitKey(1) == ord('q'):
            break
