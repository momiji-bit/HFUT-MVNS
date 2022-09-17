import matplotlib.pyplot as plt

from oak_Cam import *
from tools import *
import matplotlib
matplotlib.use('TKAgg')

if __name__ == '__main__':
    colors = Colors()
    oakCam = OakCam()
    msg = oakCam.get_msg()
    path = TrackPath(lifetime=4)
    while True:
        t0 = time.time()
        background = draw_birdView()
        RGB, D, IMU = next(msg)

        Q = Quaternion(IMU.rotationVector.real, IMU.rotationVector.i, IMU.rotationVector.j,
                       IMU.rotationVector.k)  # 旋转向量

        # RGB --目标追踪--> out1
        out1 = track(RGB)  # out1 = [[c, id, name, conf, xyxy],...]
        t2 = time.time()  # >>>TIME<<<
        # 遍历当前帧每一个bbox
        for bbox in out1:  # bbox = [c, id, name, conf, xyxy]
            color = colors(bbox[0], True)  # 一种类别对应一种颜色
            # 绘制bbox框
            cv2.rectangle(RGB, tuple(bbox[4][:2]), tuple(bbox[4][2:]), color, 1)
            lable = f'{bbox[1]} {bbox[2]}'
            cv2.putText(RGB, lable, (bbox[4][0] + 2, bbox[4][1] + 13), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

            Z_RAW, num = oakCam.getObjectDepth(D, bbox[4])  # 前+后- mm
            X_RAW = oakCam.getObjectX(Z_RAW, (bbox[4][0] + bbox[4][2]) / 2)  # 左+右- mm
            Y_RAW = oakCam.getObjectY(Z_RAW, (bbox[4][1] + bbox[4][3]) / 2)  # 上+下- mm

            World_Coordinate = Q.rotate(np.array([int(Y_RAW), int(X_RAW), int(Z_RAW)]))  # X上+下- Y左+右- Z前+后-
            # cv2.circle(background, (int(World_Coordinate[0]/10)+R, -int(World_Coordinate[1]/10)+R), radius=2,
            #            color=color, thickness=-1)
            cv2.putText(RGB, f"Z: {int(math.sqrt(int(World_Coordinate[0]+0.5) ** 2 + int(World_Coordinate[1]+0.5) ** 2))} mm",
                    (bbox[4][0] + 2, bbox[4][1] + 35), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255))

            path.updata(id=bbox[1], c=bbox[0], name=bbox[2], xyz=(int(World_Coordinate[0]), int(World_Coordinate[1]),
                                                                  int(World_Coordinate[2])), T=time.time())
            plt.hist(np.array(num), bins=10)

        points = path.get_points()
        path.small()
        for t in points:
            P = points[t]
            onePath = []
            c = None
            for point in P:
                c = point[2]
                cp = (int(point[0][0]/10)+R, -int(point[0][1]/10)+R)
                cv2.circle(background, cp, radius=2, color=colors(c, True), thickness=-1)
                onePath.append(cp)
            if len(onePath) > 1:
                cv2.polylines(background, [np.array(onePath)], isClosed=False, color=colors(c, True), thickness=1)

        lu = Q.rotate(
            np.array([1000 * math.tan(math.radians(vfov / 2)), 1000 * math.tan(math.radians(hfov / 2)), 1000]))
        ld = Q.rotate(
            np.array([-1000 * math.tan(math.radians(vfov / 2)), 1000 * math.tan(math.radians(hfov / 2)), 1000]))
        ru = Q.rotate(
            np.array([1000 * math.tan(math.radians(vfov / 2)), -1000 * math.tan(math.radians(hfov / 2)), 1000]))
        rd = Q.rotate(
            np.array([-1000 * math.tan(math.radians(vfov / 2)), -1000 * math.tan(math.radians(hfov / 2)), 1000]))
        cv2.line(background, (R, R), (int(lu[0] / 10 + 0.5) + R, - int(lu[1] / 10 + 0.5) + R), (0, 150, 0),
                 1)
        cv2.line(background, (R, R), (int(ld[0] / 10 + 0.5) + R, -int(ld[1] / 10 + 0.5) + R), (0, 0, 150),
                 1)
        cv2.line(background, (R, R), (int(ru[0] / 10 + 0.5) + R, -int(ru[1] / 10 + 0.5) + R), (0, 150, 0),
                 1)
        cv2.line(background, (R, R), (int(rd[0] / 10 + 0.5) + R, -int(rd[1] / 10 + 0.5) + R), (0, 0, 150),
                 1)

        depthDemo = cv2.applyColorMap(cv2.equalizeHist(cv2.normalize(D, None, 255, 0, cv2.NORM_INF, cv2.CV_8UC1)), cv2.COLORMAP_VIRIDIS)

        Demo = np.vstack((depthDemo, RGB))
        Demo = cv2.resize(Demo, (int(R*2*640/720+0.5), int(R*2+0.5)))
        Demo = np.hstack((background, Demo))
        cv2.putText(Demo, "FPS: {:.1f}".format(1 / (time.time() - t0)), (2, Demo.shape[0] - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)
        cv2.imshow('Demo', Demo)
        # print('done')

        plt.show()

        if cv2.waitKey(1) == ord('q'):
            break
