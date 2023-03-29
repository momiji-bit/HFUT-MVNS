# 相机姿态展示
#!/usr/bin/env python3

import cv2
import depthai as dai
import time
import math
import matplotlib.pyplot as plt
import numpy as np
from pyquaternion import Quaternion

import matplotlib
matplotlib.use('TKAgg')

from matplotlib import animation
from mpl_toolkits.mplot3d import Axes3D



pipeline = dai.Pipeline()
imu = pipeline.create(dai.node.IMU)
xlinkOut = pipeline.create(dai.node.XLinkOut)
xlinkOut.setStreamName("imu")
imu.enableIMUSensor(dai.IMUSensor.ROTATION_VECTOR, 400)
imu.setBatchReportThreshold(1)
imu.setMaxBatchReports(10)
imu.out.link(xlinkOut.input)

# imu.enableFirmwareUpdate(True)

with dai.Device(pipeline) as device:

    imuQueue = device.getOutputQueue(name="imu", maxSize=50, blocking=False)


    def generate_quaternion():
        while True:
            imuData = imuQueue.get()  # blocking call, will wait until a new data has arrived
            rVvalues = imuData.packets[-1].rotationVector
            msgs = []
            # 遍历当前帧每一个bbox
            for bbox in range(10):  # bbox = [c, id, name, conf, xyxy]
                msgs.append(bbox)
            yield msgs, Quaternion(rVvalues.real, rVvalues.i, rVvalues.j, rVvalues.k)


    quaternion_generator = generate_quaternion()

    # Set up figure & 3D axis for animation
    fig = plt.figure()
    ax = fig.add_axes([0, 0, 1, 1], projection='3d')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    # ax.axis('off')

    # use a different color for each axis
    colors = ['r', 'g', 'b', 'g', 'g', 'g', 'g']

    # set up lines and points
    lines = sum([ax.plot([], [], [], c=c)
                 for c in colors], [])

    hfov = 68.7938003540039
    vfov = 42.12409823672219
    startpoints = np.array([[-0.3, 0, 0], [0, -0.4, 0], [0, 0, -0.3],
                            [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]])
    endpoints = np.array([[0.3, 0, 0], [0, 0.4, 0], [0, 0, 1],
                          [math.tan(math.radians(vfov / 2)), math.tan(math.radians(hfov / 2)), 1],
                          [-math.tan(math.radians(vfov / 2)), math.tan(math.radians(hfov / 2)), 1],
                          [math.tan(math.radians(vfov / 2)), -math.tan(math.radians(hfov / 2)), 1],
                          [-math.tan(math.radians(vfov / 2)), -math.tan(math.radians(hfov / 2)), 1]])

    # prepare the axes limits
    ax.set_xlim((-3, 3))
    ax.set_ylim((-3, 3))
    ax.set_zlim((-3, 3))

    # set point-of-view: specified by (altitude degrees, azimuth degrees)
    ax.view_init(30, 210)


    # initialization function: plot the background of each frame
    def init():
        for line in lines:
            line.set_data([], [])
            line.set_3d_properties([])

        return lines


    # animation function.  This will be called sequentially with the frame number
    def animate(i):
        # we'll step two time-steps per frame.  This leads to nice results.
        # i = (2 * i) % x_t.shape[1]

        msg, q = next(quaternion_generator)
        print("q:", q)

        for line, start, end in zip(lines, startpoints, endpoints):
            # end *= 5
            start = q.rotate(start)
            end = q.rotate(end)

            line.set_data([start[0], end[0]], [start[1], end[1]])
            line.set_3d_properties([start[2], end[2]])

            # pt.set_data(x[-1:], y[-1:])
            # pt.set_3d_properties(z[-1:])

        # ax.view_init(30, 0.6 * i)
        fig.canvas.draw()
        return lines


    # instantiate the animator.
    anim = animation.FuncAnimation(fig, animate, init_func=init,
                                   frames=500, interval=30, blit=True)

    # Save as mp4. This requires mplayer or ffmpeg to be installed
    # anim.save('lorentz_attractor.mp4', fps=15, extra_args=['-vcodec', 'libx264'])

    plt.show()

