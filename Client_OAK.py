# 矫正后的彩色画面与深度图，边缘重合

import cv2
import depthai as dai
import numpy as np
import time
import socket
import cv2
import numpy
import time
import sys

def getFramess(queue):
    """获取队列中最后一帧画面"""
    # Get frame from queue
    frame = queue.get()
    # Convert frame to OpenCV format and return
    return frame.getCvFrame()


def to_COLORMAP(depth):
    depth = cv2.normalize(depth, None, 255, 0, cv2.NORM_MINMAX, cv2.CV_8UC1)
    depth = cv2.equalizeHist(depth)
    return cv2.applyColorMap(255 - depth, cv2.COLORMAP_HOT)


def recvall(sock, count):
    buf = b''  # buf是一个byte类型
    while count:
        # 接受TCP套接字的数据。数据以字符串形式返回，count指定要接收的最大数据量.
        newbuf = sock.recv(count)
        if not newbuf: return None
        buf += newbuf
        count -= len(newbuf)
    return buf


if __name__ == '__main__':
    lrcheck  = True   # Better handling for occlusions
    extended = False  # Closer-in minimum depth, disparity range is doubled
    subpixel = True   # Better accuracy for longer distance, fractional disparity 32-levels
    # Options: MEDIAN_OFF, KERNEL_3x3, KERNEL_5x5, KERNEL_7x7
    median   = dai.StereoDepthProperties.MedianFilter.KERNEL_3x3

    # print("StereoDepth config options:")
    # print("    Left-Right check:  ", lrcheck)
    # print("    Extended disparity:", extended)
    # print("    Subpixel:          ", subpixel)
    # print("    Median filtering:  ", median)

    pipeline = dai.Pipeline()

    monoLeft = pipeline.create(dai.node.MonoCamera)
    monoLeft.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
    monoLeft.setBoardSocket(dai.CameraBoardSocket.LEFT)
    # monoLeft.setFps(30)

    monoRight = pipeline.create(dai.node.MonoCamera)
    monoRight.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
    monoRight.setBoardSocket(dai.CameraBoardSocket.RIGHT)
    # monoRight.setFps(30)

    stereo = pipeline.createStereoDepth()
    stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)
    stereo.initialConfig.setMedianFilter(median)
    stereo.setLeftRightCheck(lrcheck)
    stereo.setExtendedDisparity(extended)
    stereo.setSubpixel(subpixel)
    monoLeft.out.link(stereo.left)
    monoRight.out.link(stereo.right)

    config = stereo.initialConfig.get()
    config.postProcessing.speckleFilter.enable = False
    config.postProcessing.speckleFilter.speckleRange = 50
    config.postProcessing.temporalFilter.enable = True
    config.postProcessing.spatialFilter.enable = True
    config.postProcessing.spatialFilter.holeFillingRadius = 2
    config.postProcessing.spatialFilter.numIterations = 1
    config.postProcessing.thresholdFilter.minRange = 0  # 0.2m
    config.postProcessing.thresholdFilter.maxRange = 20000  # 20m
    config.postProcessing.decimationFilter.decimationFactor = 1
    stereo.initialConfig.set(config)

    xout_depth = pipeline.createXLinkOut()
    xout_depth.setStreamName('depth')
    stereo.depth.link(xout_depth.input)

    # xout_disparity = pipeline.createXLinkOut()
    # xout_disparity.setStreamName('disparity')
    # stereo.disparity.link(xout_disparity.input)

    xout_colorize = pipeline.createXLinkOut()
    xout_colorize.setStreamName('colorize')

    camRgb = pipeline.create(dai.node.ColorCamera)
    camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
    camRgb.setIspScale(1, 3)
    camRgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.RGB)
    camRgb.initialControl.setManualFocus(130)
    stereo.setDepthAlign(dai.CameraBoardSocket.RGB)
    camRgb.isp.link(xout_colorize.input)
    # camRgb.setFps(30)

    class HostSync:
        def __init__(self):
            self.arrays = {}

        def add_msg(self, name, msg):
            if not name in self.arrays:
                self.arrays[name] = []
            # Add msg to array
            self.arrays[name].append({'msg': msg, 'seq': msg.getSequenceNum()})

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


    with dai.Device(pipeline) as device:

        device.setIrLaserDotProjectorBrightness(1200)
        qs = []
        qs.append(device.getOutputQueue("depth", 1))
        qs.append(device.getOutputQueue("colorize", 1))

        calibData = device.readCalibration()
        w, h = camRgb.getIspSize()
        intrinsics = calibData.getCameraIntrinsics(dai.CameraBoardSocket.RGB, dai.Size2f(w, h))

        sync = HostSync()
        # TCP连接
        address = ('172.16.36.120', 8888)
        # 3.tcp.cpolar.top 11670
        try:
            # 建立socket对象，参数意义见https://blog.csdn.net/rebelqsp/article/details/22109925
            # socket.AF_INET：服务器之间网络通信
            # socket.SOCK_STREAM：流式socket , for TCP
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            # 开启连接
            sock.connect(address)
        except socket.error as msg:
            print(msg)
            sys.exit(1)
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 95]  # 压缩会提高帧率，但会更占用CPU以及更长的延时

        while True:
            t0 = time.time()
            for q in qs:
                new_msg = q.tryGet()
                if new_msg is not None:
                    msgs = sync.add_msg(q.getName(), new_msg)
                    if msgs:
                        color = msgs["colorize"].getCvFrame()
                        result_RGB, imgencode_RGB = cv2.imencode('.jpg', color, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
                        data_RGB = numpy.array(imgencode_RGB)
                        stringData_RGB = data_RGB.tobytes()

                        depth = msgs["depth"].getFrame()
                        # depth = to_COLORMAP(depth)
                        depth_RGB = (cv2.cvtColor(depth, cv2.COLOR_GRAY2BGR)/20000)*255
                        depth_RGB = depth_RGB.astype(np.uint8)
                        result_D, imgencode_D = cv2.imencode('.jpg', depth_RGB, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
                        data_D = numpy.array(imgencode_D)
                        stringData_D = data_D.tobytes()

                        sock.send(str.encode(str(len(stringData_RGB)).ljust(16)))
                        sock.send(str.encode(str(len(stringData_D)).ljust(16)))
                        sock.send(stringData_RGB)
                        sock.send(stringData_D)

                        echo = recvall(sock, 7).decode('UTF-8')
                        if echo == 'GUJIHAO':
                            print(f'\rtime={(time.time()-t0)*1000:.1f}ms', end='')

                        # print(int((time.time()-t0)*1000), 'ms')
                        # cv2.imshow('a', depth_RGB)

                        if cv2.waitKey(1) == ord('q'):
                            break

