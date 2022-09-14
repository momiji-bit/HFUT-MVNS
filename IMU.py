#!/usr/bin/env python3

import cv2
import depthai as dai
import time
import math

import matplotlib.pyplot as plt

# Create pipeline
pipeline = dai.Pipeline()

# Define sources and outputs
imu = pipeline.create(dai.node.IMU)
xlinkOut = pipeline.create(dai.node.XLinkOut)

xlinkOut.setStreamName("imu")

# enable ACCELEROMETER_RAW at 500 hz rate
imu.enableIMUSensor(dai.IMUSensor.LINEAR_ACCELERATION, 500)
# enable GYROSCOPE_RAW at 400 hz rate
imu.enableIMUSensor(dai.IMUSensor.MAGNETOMETER_CALIBRATED, 500)
imu.enableIMUSensor(dai.IMUSensor.ROTATION_VECTOR, 500)
# it's recommended to set both setBatchReportThreshold and setMaxBatchReports to 20 when integrating in a pipeline with a lot of input/output connections
# above this threshold packets will be sent in batch of X, if the host is not blocked and USB bandwidth is available
imu.setBatchReportThreshold(1)
# maximum number of IMU packets in a batch, if it's reached device will block sending until host can receive it
# if lower or equal to batchReportThreshold then the sending is always blocking on device
# useful to reduce device's CPU load  and number of lost packets, if CPU load is high on device side due to multiple nodes
imu.setMaxBatchReports(10)

# Link plugins IMU -> XLINK
imu.out.link(xlinkOut.input)

# Pipeline is defined, now we can connect to the device
with dai.Device(pipeline) as device:

    def timeDeltaToMilliS(delta) -> float:
        return delta.total_seconds()*1000

    # Output queue for imu bulk packets
    imuQueue = device.getOutputQueue(name="imu", maxSize=50, blocking=False)
    baseTs = None

    ta = []
    a = []
    tg = []
    g = []
    while True:
        imuData = imuQueue.get()  # blocking call, will wait until a new data has arrived

        imuPackets = imuData.packets
        for imuPacket in imuPackets:
            acceleroValues = imuPacket.acceleroMeter  # 加速度
            magneticField = imuPacket.magneticField  # 磁力
            rotationVector = imuPacket.rotationVector  # 陀螺

            # acceleroTs = acceleroValues.timestamp.get()
            # magneticTs = magneticField.timestamp.get()
            # rotationTs = rotationVector.timestamp.get()
            # if baseTs is None:
            #     baseTs = acceleroTs if acceleroTs < rotationTs else rotationTs
            # acceleroTs = timeDeltaToMilliS(acceleroTs - baseTs)
            # magneticTs = timeDeltaToMilliS(magneticTs - baseTs)
            # rotationTs = timeDeltaToMilliS(rotationTs - baseTs)

            # ta.append(acceleroTs)
            # a.append(math.sqrt(magneticTs.x**2+magneticTs.y**2+magneticTs.z**2))
            # tg.append(rotationTs)
            # g.append(math.sqrt(gyroValues.x**2+gyroValues.y**2+gyroValues.z**2))
            # print(f"{rotationVector.real}\t\t{rotationVector.i}\t\t{rotationVector.j}\t\t{rotationVector.k}")
            print(f"{rotationVector.i*rotationVector.i}")
            # print(f"{math.sqrt(magneticField.x**2+magneticField.y**2+magneticField.z**2)}")
            # print(f"Accelerometer [m/s^2]: x: {imuF.format(acceleroValues.x)} y: {imu F.format(acceleroValues.y)} z: {imuF.format(acceleroValues.z)}")
            # print(f"Gyroscope timestamp: {tsF.format(gyroTs)} ms")
            # print(f"Gyroscope [rad/s]: x: {imuF.format(gyroValues.x)} y: {imuF.format(gyroValues.y)} z: {imuF.format(gyroValues.z)} ")

        if cv2.waitKey(1) == ord('q'):
            break
        if len(a)>500:
            break

    plt.plot(ta, a)
    # plt.plot(tg, g)
    plt.savefig('out.png')