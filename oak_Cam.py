import cv2
import depthai as dai
import numpy as np
import math
import matplotlib
import matplotlib.pyplot as plt

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
            print(f'\rname: {name}, seq: {msg.getSequenceNum()}', end='')

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
        print('OAK preparing...')
        self.lrcheck = True
        self.extended = False
        self.subpixel = True
        self.median = dai.StereoDepthProperties.MedianFilter.KERNEL_3x3

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
        # 初始化管道
        pipeline = dai.Pipeline()

        # 定义IMU
        imu = pipeline.create(dai.node.IMU)
        xout_imu = pipeline.create(dai.node.XLinkOut)
        xout_imu.setStreamName("IMU_Stream")
        imu.enableIMUSensor(dai.IMUSensor.ROTATION_VECTOR, 1000)
        imu.setBatchReportThreshold(1)
        imu.setMaxBatchReports(10)
        imu.out.link(xout_imu.input)

        # 左黑白相机
        monoLeft = pipeline.create(dai.node.MonoCamera)
        monoLeft.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)  # 分辨率
        monoLeft.setBoardSocket(dai.CameraBoardSocket.LEFT)
        # 右黑白相机
        monoRight = pipeline.create(dai.node.MonoCamera)
        monoRight.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
        monoRight.setBoardSocket(dai.CameraBoardSocket.RIGHT)
        # 立体匹配
        stereo = pipeline.createStereoDepth()
        stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)
        stereo.initialConfig.setMedianFilter(self.median)  # MEDIAN_OFF, KERNEL_3x3, KERNEL_5x5, KERNEL_7x7
        stereo.setLeftRightCheck(self.lrcheck)  # 更好地处理遮挡
        stereo.setExtendedDisparity(self.extended)  # 越近最小深度，视差范围翻倍
        stereo.setSubpixel(self.subpixel)  # 更远距离的精度更高，部分差异 32 级
        monoLeft.out.link(stereo.left)  # 左右黑白相机数据接入
        monoRight.out.link(stereo.right)
        # 立体匹配参数
        config = stereo.initialConfig.get()
        config.postProcessing.speckleFilter.enable = False
        config.postProcessing.speckleFilter.speckleRange = 50
        config.postProcessing.temporalFilter.enable = True
        config.postProcessing.spatialFilter.enable = True
        config.postProcessing.spatialFilter.holeFillingRadius = 2
        config.postProcessing.spatialFilter.numIterations = 1
        config.postProcessing.thresholdFilter.minRange = 200  # 0.2m
        config.postProcessing.thresholdFilter.maxRange = 10000  # 20m
        config.postProcessing.decimationFilter.decimationFactor = 1
        stereo.initialConfig.set(config)
        # 深度图输出流
        xout_depth = pipeline.createXLinkOut()
        xout_depth.setStreamName('D_Stream')  # 深度视频流
        stereo.depth.link(xout_depth.input)
        # 彩色图输出流，与深度图对齐
        xout_colorize = pipeline.createXLinkOut()
        xout_colorize.setStreamName('RGB_Stream')  # 彩色视频流
        camRgb = pipeline.create(dai.node.ColorCamera)
        camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
        camRgb.setIspScale(1, 3)  # 设置“isp”输出比例（分子/分母）
        camRgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.RGB)
        camRgb.initialControl.setManualFocus(130)  # 设置手动对焦位置
        stereo.setDepthAlign(dai.CameraBoardSocket.RGB)  # 设置将对齐视差/深度的视角的相机 *
        camRgb.isp.link(xout_colorize.input)

        with dai.Device(pipeline) as device:
            sync = self.HostSync()  # tongbuqi
            qs = []
            qs.append(device.getOutputQueue("D_Stream", 5, False))
            qs.append(device.getOutputQueue("RGB_Stream", 5, False))
            IMU_Stream_Queue = device.getOutputQueue("IMU_Stream", 120, False)

            while True:
                for q in qs:
                    new_msg = q.tryGet()
                    if new_msg is not None:
                        msgs = sync.add_msg(q.getName(), new_msg)
                        if msgs:
                            D = msgs["D_Stream"].getFrame()  # 原始深度图
                            RGB = msgs["RGB_Stream"].getCvFrame()
                            IMU = IMU_Stream_Queue.get().packets[0]
                            # print(f'\rseq: {IMU.rotationVector.sequence}', end='')
                            yield RGB, D, IMU


if __name__ == '__main__':
    oakCam = OakCam()
    msg = oakCam.get_msg()
    while True:
        RGB, D, IMU = next(msg)
        cv2.imshow('RGB', RGB)
        cv2.imshow('D', D)
        if cv2.waitKey(1) == ord('q'):
            break
