# 矫正后的彩色画面与深度图，边缘重合

import cv2
import depthai as dai
import numpy as np
import time


lrcheck  = True   # Better handling for occlusions
extended = False  # Closer-in minimum depth, disparity range is doubled
subpixel = True   # Better accuracy for longer distance, fractional disparity 32-levels
# Options: MEDIAN_OFF, KERNEL_3x3, KERNEL_5x5, KERNEL_7x7
median   = dai.StereoDepthProperties.MedianFilter.KERNEL_3x3

print("StereoDepth config options:")
print("    Left-Right check:  ", lrcheck)
print("    Extended disparity:", extended)
print("    Subpixel:          ", subpixel)
print("    Median filtering:  ", median)

pipeline = dai.Pipeline()

monoLeft = pipeline.create(dai.node.MonoCamera)
monoLeft.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
monoLeft.setBoardSocket(dai.CameraBoardSocket.LEFT)

monoRight = pipeline.create(dai.node.MonoCamera)
monoRight.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
monoRight.setBoardSocket(dai.CameraBoardSocket.RIGHT)

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
config.postProcessing.thresholdFilter.minRange = 200  # 0.2m
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
        if len(synced) == 2: # color, depth, nn
            # Remove old msgs
            for name, arr in self.arrays.items():
                for i, obj in enumerate(arr):
                    if obj['seq'] < msg.getSequenceNum():
                        arr.remove(obj)
                    else: break
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


    save = None
    f = True
    while True:
        t0 = time.time()
        for q in qs:
            new_msg = q.tryGet()
            if new_msg is not None:
                msgs = sync.add_msg(q.getName(), new_msg)
                if msgs:
                    depth = msgs["depth"].getFrame()
                    color = msgs["colorize"].getCvFrame()

                    depthDemo = cv2.normalize(depth, None, 255, 0, cv2.NORM_INF, cv2.CV_8UC1)
                    depthDemo = cv2.equalizeHist(depthDemo)
                    depthDemo = cv2.applyColorMap(255 - depthDemo, cv2.COLORMAP_OCEAN)
                    # out = np.uint8(color * 2 / 3 + depthDemo / 3)
                    cv2.imshow('color', color)
                    cv2.imshow('depthDemo', depthDemo)
                    #
                    # cv2.putText(out, "fps: {:}".format(int(1 / (time.time() - t0))), (2, out.shape[0] - 4),
                    #             cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1)
                    # cv2.imshow('out', out)


        if cv2.waitKey(1) == ord('q'):
            break

