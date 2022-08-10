#!/usr/bin/env python3

import cv2
import depthai as dai

def point_to_ConfigData(topLeft_px, bottomRight_px, imgSize_wh=[400,640], lowerThreshold=0, upperThreshold=20000):
    topLeft = dai.Point2f(topLeft_px[0]/imgSize_wh[0], topLeft_px[1]/imgSize_wh[1])
    bottomRight = dai.Point2f(bottomRight_px[0]/imgSize_wh[0], bottomRight_px[1]/imgSize_wh[1])
    config = dai.SpatialLocationCalculatorConfigData()
    config.depthThresholds.lowerThreshold = lowerThreshold
    config.depthThresholds.upperThreshold = upperThreshold
    config.roi = dai.Rect(topLeft, bottomRight)
    return config

stepSize = 0.05

# 开始定义管道
pipeline = dai.Pipeline()

# 定义双目相机
monoLeft = pipeline.createMonoCamera()
monoRight = pipeline.createMonoCamera()
stereo = pipeline.createStereoDepth()
spatialLocationCalculator = pipeline.createSpatialLocationCalculator()

xoutDepth = pipeline.createXLinkOut()
xoutSpatialData = pipeline.createXLinkOut()
xinSpatialCalcConfig = pipeline.createXLinkIn()

xoutDepth.setStreamName("depth")
xoutSpatialData.setStreamName("spatialData")
xinSpatialCalcConfig.setStreamName("spatialCalcConfig")

# 双目相机
monoLeft.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
monoLeft.setBoardSocket(dai.CameraBoardSocket.LEFT)
monoRight.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
monoRight.setBoardSocket(dai.CameraBoardSocket.RIGHT)

lrcheck = True
subpixel = False

# 双目深度
stereo.setConfidenceThreshold(255)

stereo.setLeftRightCheck(lrcheck)
stereo.setSubpixel(subpixel)

monoLeft.out.link(stereo.left)
monoRight.out.link(stereo.right)

spatialLocationCalculator.passthroughDepth.link(xoutDepth.input)
stereo.depth.link(spatialLocationCalculator.inputDepth)

spatialLocationCalculator.setWaitForConfigInput(False)


# config = dai.SpatialLocationCalculatorConfigData()
# config.depthThresholds.lowerThreshold = 100
# config.depthThresholds.upperThreshold = 10000
# spatialLocationCalculator.initialConfig.addROI(config)
spatialLocationCalculator.out.link(xoutSpatialData.input)
xinSpatialCalcConfig.out.link(spatialLocationCalculator.inputConfig)


# 连接并开启管道
with dai.Device(pipeline) as device:

    # 输出队列将用于从上面定义的输出中获取深度帧
    depthQueue = device.getOutputQueue(name="depth", maxSize=4, blocking=False)
    spatialCalcQueue = device.getOutputQueue(name="spatialData", maxSize=4, blocking=False)
    spatialCalcConfigInQueue = device.getInputQueue("spatialCalcConfig")

    color = (255, 255, 255)

    print("Use WASD keys to move ROI!")

    while True:


        cfg = dai.SpatialLocationCalculatorConfig()
        for p in [[1,1,10,10],[11,11,40,40],[80,80,300,300]]:
            c = point_to_ConfigData(p[:2], p[2:])
            cfg.addROI(c)
        spatialCalcConfigInQueue.send(cfg)


        inDepth = depthQueue.get() # 阻止通话，将等待直到新数据到达
        inDepthAvg = spatialCalcQueue.get() # 阻止通话，将等待直到新数据到达

        depthFrame = inDepth.getFrame()
        depthFrameColor = cv2.normalize(depthFrame, None, 255, 0, cv2.NORM_INF, cv2.CV_8UC1)
        depthFrameColor = cv2.equalizeHist(depthFrameColor)
        depthFrameColor = cv2.applyColorMap(255 - depthFrameColor, cv2.COLORMAP_HOT)



        spatialData = inDepthAvg.getSpatialLocations()
        for depthData in spatialData:
            roi = depthData.config.roi
            roi = roi.denormalize(width=depthFrameColor.shape[1], height=depthFrameColor.shape[0])
            xmin = int(roi.topLeft().x)
            ymin = int(roi.topLeft().y)
            xmax = int(roi.bottomRight().x)
            ymax = int(roi.bottomRight().y)

            fontType = cv2.FONT_HERSHEY_TRIPLEX
            cv2.rectangle(depthFrameColor, (xmin, ymin), (xmax, ymax), color, cv2.FONT_HERSHEY_SCRIPT_SIMPLEX)
            cv2.putText(depthFrameColor, f"X: {int(depthData.spatialCoordinates.x)} mm", (xmin + 10, ymin + 20), fontType, 0.5, color)
            cv2.putText(depthFrameColor, f"Y: {int(depthData.spatialCoordinates.y)} mm", (xmin + 10, ymin + 35), fontType, 0.5, color)
            cv2.putText(depthFrameColor, f"Z: {int(depthData.spatialCoordinates.z)} mm", (xmin + 10, ymin + 50), fontType, 0.5, color)

        cv2.imshow("depth", depthFrameColor)
        if cv2.waitKey(1) == ord('q'):
            break



