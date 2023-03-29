import depthai as dai
import numpy as np
import cv2
import time

# Create pipeline
pipeline = dai.Pipeline()

# This might improve reducing the latency on some systems
pipeline.setXLinkChunkSize(0)

# Define source and output
camRgb = pipeline.create(dai.node.ColorCamera)
camRgb.setFps(30)
camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_720_P)
camRgb.setIspScale(2, 3)

stereo = pipeline.create(dai.node.StereoDepth)
stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)
stereo.initialConfig.setMedianFilter(dai.MedianFilter.KERNEL_7x7)
stereo.setLeftRightCheck(True)
stereo.setDepthAlign(dai.CameraBoardSocket.RGB)
stereo.setExtendedDisparity(False)
stereo.setSubpixel(False)

camMNR = pipeline.create(dai.node.MonoCamera)
camMNR.setFps(30)
camMNR.setResolution(dai.MonoCameraProperties.SensorResolution.THE_720_P)
camMNR.setBoardSocket(dai.CameraBoardSocket.RIGHT)
camMNR.out.link(stereo.right)

camMNL = pipeline.create(dai.node.MonoCamera)
camMNL.setFps(30)
camMNL.setResolution(dai.MonoCameraProperties.SensorResolution.THE_720_P)
camMNL.setBoardSocket(dai.CameraBoardSocket.LEFT)
camMNL.out.link(stereo.left)

imu = pipeline.create(dai.node.IMU)
imu.enableIMUSensor([dai.IMUSensor.ACCELEROMETER_RAW, dai.IMUSensor.GYROSCOPE_RAW], 100)
imu.setBatchReportThreshold(1)
imu.setMaxBatchReports(10)

xoutRGB = pipeline.create(dai.node.XLinkOut)
xoutRGB.setStreamName("camRgb")
camRgb.isp.link(xoutRGB.input)

xoutD = pipeline.create(dai.node.XLinkOut)
xoutD.setStreamName("Depth")
stereo.depth.link(xoutD.input)

xoutIMU = pipeline.create(dai.node.XLinkOut)
xoutIMU.setStreamName("IMU")
imu.out.link(xoutIMU.input)

# xoutMNR = pipeline.create(dai.node.XLinkOut)
# xoutMNR.setStreamName("camMNR")
# camMNR.out.link(xoutMNR.input)

# Connect to device and start pipeline
with dai.Device(pipeline) as device:
    device.setIrLaserDotProjectorBrightness(0)  # in mA, 0..1200
    device.setIrFloodLightBrightness(0)  # in mA, 0..1500

    q_camRgb = device.getOutputQueue(name="camRgb", maxSize=1, blocking=False)
    q_Depth = device.getOutputQueue(name="Depth", maxSize=1, blocking=False)
    q_IMU = device.getOutputQueue(name="IMU", maxSize=1, blocking=False)

    rgbWeight = 0.4
    depthWeight = 0.6

    def updateBlendWeights(percent_rgb):
        """
        Update the rgb and depth weights used to blend depth/rgb image

        @param[in] percent_rgb The rgb weight expressed as a percentage (0..100)
        """
        global depthWeight
        global rgbWeight
        rgbWeight = float(percent_rgb) / 100.0
        depthWeight = 1.0 - rgbWeight


    frameRgb = None
    frameDisp = None
    blendedWindowName = "rgb-depth"
    cv2.namedWindow(blendedWindowName)
    cv2.createTrackbar('RGB Weight %', blendedWindowName, int(rgbWeight * 100), 100, updateBlendWeights)

    fpsRgb = []
    fpsD = []
    fps = []
    latencySRGB = None
    latencySD = None
    while True:
        t0 = time.time()
        imgRgb = q_camRgb.get()
        Depth = q_Depth.get()
        IMU = q_IMU.get()

        frameRgb = imgRgb.getCvFrame()
        latencySRGB = (dai.Clock.now() - imgRgb.getTimestamp()).total_seconds()
        fpsRgb.append(1 / latencySRGB)
        fpsRgb = fpsRgb[-120:]

        frameDisp = Depth.getFrame()
        frameDisp = cv2.applyColorMap(
            cv2.equalizeHist(cv2.normalize(-frameDisp, None, 255, 0, cv2.NORM_INF, cv2.CV_8UC1)), cv2.COLORMAP_JET)
        latencySD = (dai.Clock.now() - Depth.getTimestamp()).total_seconds()
        fpsD.append(1 / latencySD)
        fpsD = fpsD[-120:]

        if frameRgb is not None and frameDisp is not None:
            blended = cv2.addWeighted(frameRgb, rgbWeight, frameDisp, depthWeight, 0)
            cv2.putText(blended,
                        "RGB FPS: {:.1f} | Latency: {:.2f} ms".format(sum(fpsRgb) / max(1, len(fpsRgb)), latencySRGB * 1000),
                        (2, frameDisp.shape[0] - 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)
            cv2.putText(blended,
                        "DEP FPS: {:.1f} | Latency: {:.2f} ms".format(sum(fpsD) / max(1, len(fpsD)), latencySD * 1000),
                        (2, frameRgb.shape[0] - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)
            cv2.imshow(blendedWindowName, blended)
            frameRgb = None
            frameDisp = None

        if cv2.waitKey(1) == ord('q'):
            break
        # if time.time() > t0:
        #     fps.append(1 / (time.time()-t0))
        #     fps = fps[-120:]

