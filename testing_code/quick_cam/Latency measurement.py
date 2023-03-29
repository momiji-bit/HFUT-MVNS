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
camRgb.setFps(60)
camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)

xout = pipeline.create(dai.node.XLinkOut)
xout.setStreamName("out")
camRgb.isp.link(xout.input)

# Connect to device and start pipeline
with dai.Device(pipeline) as device:
    q = device.getOutputQueue(name="out", maxSize=1, blocking=False)
    diffs = np.array([])
    fps = []
    while True:
        t0 = time.time()
        imgFrame = q.tryGet()
        if imgFrame:
            # Latency in miliseconds
            latencyMs = (dai.Clock.now() - imgFrame.getTimestamp()).total_seconds() * 1000
            diffs = np.append(diffs, latencyMs)
            print('\rLatency: {:.2f} ms, Average latency: {:.2f} ms, Std: {:.2f}'.format(latencyMs, np.average(diffs),
                                                                                         np.std(diffs)), end='')
            out = imgFrame.getCvFrame()
            t = time.time() - t0
            if t != 0:
                fps.append(1 / t)
                fps = fps[-120:]
            cv2.putText(out, "FPS: {:.1f}".format(sum(fps) / len(fps)), (2, out.shape[0] - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)
            cv2.imshow('frame', out)
            if cv2.waitKey(1) == ord('q'):
                break
