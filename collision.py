import math
import random

import cv2
import numpy as np
import torch

from scipy.stats import spearmanr, kendalltau
# from preparation import *
from yolov5.utils.plots import colors

class Kfilter:
    def __init__(self):

        self.kf = cv2.KalmanFilter(4, 4 ,0)
        self.kf.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0],[0,0,1,0],[0,0,0,1]], np.float32)
        self.kf.transitionMatrix = np.array([[1, 0, 0.2, 0], [0, 1, 0, 0.2], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)

    def predict(self, coordX, coordY, vx , vy):
        '''This function estimates the position of the object'''
        measured = np.array([[np.float32(coordX)], [np.float32(coordY)], [np.float32(vx)], [np.float32(vy)]])
        print('original',measured)
        self.kf.correct(measured)
        predicted = self.kf.predict()
        x, y = int(predicted[0]), int(predicted[1])
        print('ans point',x,y)
        return x, y


class simplePredict:
    '''
    一个简单的思路, 可以使用前一段时间内的每一段时间的速度的加权平均值, 得到一个平均的速度作为最终的速度
    并且也把每一个路线单独的绘制出来
    置信度的计算: 使用每一个时间距离开始的那一个时间的距离占总时间的比值作为置信度.
    x_{k + 1} = F \times x_{k}
    其中
    x_{k} = [x,y,vx,vy]^T
    '''

    def __init__(self):
        self.transitionMatrix = np.array([[1, 0, 0.2, 0], [0, 1, 0, 0.2], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)

    def predict(self, x, y , vx, vy):
        measured = np.array([[np.float32(x)], [np.float32(y)], [np.float32(vx)], [np.float32(vy)]])
        position = self.transitionMatrix @ measured
        return int(position[0][0]),int(position[1][0])

def draw_birdView():
    fov = 68.7938
    frame = np.zeros((1600, 1000, 3), np.uint8)
    alpha = (180 - fov) / 2
    center = int(frame.shape[1] / 2)
    max_p = frame.shape[0] - int(math.tan(math.radians(alpha)) * center)
    fov_cnt = np.array([
        (0, frame.shape[0]),
        (frame.shape[1], frame.shape[0]),
        (frame.shape[1], max_p),
        (center, frame.shape[0]),
        (0, max_p),
        (0, frame.shape[0]),
    ])
    cv2.fillPoly(frame, [fov_cnt], color=(70, 70, 70))

    fontType = cv2.FONT_HERSHEY_TRIPLEX
    color = (255,255,255)
    cv2.putText(frame, '-0M', (0, 1600), fontType, 0.8, color, 1)
    cv2.putText(frame, '-1M', (0, 1500), fontType, 0.8, color, 1)
    cv2.putText(frame, '-2M', (0, 1400), fontType, 0.8, color, 1)
    cv2.putText(frame, '-3M', (0, 1300), fontType, 0.8, color, 1)
    cv2.putText(frame, '-5M', (0, 1100), fontType, 0.8, color, 1)
    cv2.putText(frame, '-9M', (0, 700), fontType, 0.8, color, 1)
    cv2.putText(frame, '-15M', (0, 100), fontType, 0.8, color, 1)

    return frame

class collision_detection:
    def __init__(self):
        '''
            列表中每一个变量的含义如下
            # name : 类别名
            # dists : 存放 fps * time 帧图片中某物体与相机之间的距离，单位为 m
            # points : 存放 fps * time 帧图片中某物体中心坐标点的位置
            # times : 存放 fps * time 帧图片中记录某物体状态的时间，单位为 s
            # count : 记录连续命中的次数
            # 记录这一帧是否含有这个物体
            # 记录age
            # xyz : 记录真实世界下的坐标.
        '''
        # self.maxage = cfg.STRONGSORT.MAX_AGE
        self. maxage = 30
        self.dataOfobject = [[None, [], [], [], 0 , False ,self.maxage, [] ] for i in range(1000)]
        self.clips = 10
        self.background = draw_birdView()
        # self.kf = Kfilter()
        #定义一个简单的预测器
        self.sp = simplePredict()

    def predict_position(self,packOfPoints,packOfT):
        predict_point = []
        # print('packOfpoint',packOfPoints)

        i = 0
        while i < len(packOfPoints) - 1:
            #如果时间间隔过小, 就认为是同一个点的抖动
            # t_s = packOfT[i]
            # x_mean,y_mean,x_s,y_s = 0,0,packOfPoints[i][0],0
            # while i < len(packOfPoints) - 1 and packOfT[i+1] - t_s < 0.1:
            #     x_s +=

            vx = (packOfPoints[i+1][0] - packOfPoints[i][0]) / (packOfT[i+1] - packOfT[i])
            vy = (packOfPoints[i+1][1] - packOfPoints[i][1]) / (packOfT[i+1] - packOfT[i])

            x,y = self.sp.predict(packOfPoints[-1][0],packOfPoints[-1][1],vx,vy)
            predict_point.append((x,y))
            i += 1
        # print('pred',predict_point)

        return predict_point

    def update_object(self,id,dist,point,name,time,xyz):
        '''
            #在更新数据的时候，不仅仅要把相应的信息传进去，而且需要设置这个物体在这一帧画面中出现过
        '''
        self.dataOfobject[id][0] = name
        self.dataOfobject[id][1].append(dist)
        self.dataOfobject[id][2].append(point)
        self.dataOfobject[id][3].append(time)
        self.dataOfobject[id][5] = True
        self.dataOfobject[id][7].append(xyz)

        if len(self.dataOfobject[id][2]) > self.clips * 2:
            self.clear(id)


    def update_clips(self,fps,time):
        self.clips = fps * time

    def update(self):
        '''
            #更新每一个物体的状态
            #通过遍历所有的在检测器之中的物体
            #首先应该查看这些物体的标志，查看他们是否是在这一帧图片之中出现过，如果没有，就将他们的age自减1
            #当age自减为0的时候，表示这一个物体已经从视野里面消失了，可以将这一个单元的空间释放
        '''
        max_len = len(self.dataOfobject)
        id = 0
        while id < max_len:
            if not self.dataOfobject[id][5] and self.dataOfobject[id][0]:
                # print('这个物体没有在这一帧出现过')
                self.dataOfobject[id][6] -= 1
            elif self.dataOfobject[id][0]:
                # print('这个物体出现过，我们把他的标志位复原，他是',self.dataOfobject[id][0])
                self.dataOfobject[id][5] = False

            if self.dataOfobject[id][6] == 0:
                # print('有一个物体被清理了，他是',self.dataOfobject[id][0])
                self.dataOfobject[id] = [None, [], [], [], 0 , False ,self.maxage,[]]
            id += 1

    def is_near(self,id):
        '''
            本函数判断物体是否靠近
            当t和d之间为负相关的时候，表示相机和物体相对靠近
            考虑两种不同的相关系数检测方式
            1. kendall  相关系数
            2. Spearman 相关系数
            3. k        使用斜率
        '''
        name, dists, points, times, resOfcc, falg ,age, xyz = self.dataOfobject[id]
        if len(dists) > self.clips :
            theta0,_ = np.polyfit(times[-self.clips:], dists[-self.clips:], 1)
            pt = np.array(times[-self.clips:])
            pd = np.array(dists[-self.clips:])
            r_p = spearmanr(pt,pd)
            r_k = kendalltau(pt,pd)
            # if name == 'person':
                # print(r_p[0])
                # print(r_k[0])
            # if theta0 < -0.4 : #可以认为是行人正在靠近
                # print(theta0,f'{name} 靠近中')
                # print(dists[-self.clips:])
            if r_p[0] < -0.7:
                return True
            else:
                return False

    def is_direction(self,id):
        '''
            当时间t和坐标点x位置有强烈的相关关系的时候，表示物体的运动轨迹和相机不共线
            当时间t和坐标x的位置相关性较弱的时候，表示物体的运动轨迹和相机共线
        '''
        name, dists, points, times, resOfcc, falg, age, xyz = self.dataOfobject[id]
        if len(points) > self.clips:
            # pt = np.array(times)
            px = np.array(points)[-self.clips:, 0]
            # varst = np.var(pt)
            varsx = np.var(px)
            # r = (np.mean((pt * px)) - np.mean(pt) * np.mean(px)) / np.sqrt(varst * varsx)
            # if name == 'person':
            #     print(varsx)
            if varsx < 50:
                return True
            else :
                return False
    def clear(self,id):
        '''
        这个函数把所有的过量的数据都删除
        '''
        self.dataOfobject[id][1] = self.dataOfobject[id][1][-int(self.clips * 1.5):]
        self.dataOfobject[id][2] = self.dataOfobject[id][2][-int(self.clips * 1.5):]
        self.dataOfobject[id][3] = self.dataOfobject[id][3][-int(self.clips * 1.5):]
        self.dataOfobject[id][7] = self.dataOfobject[id][7][-int(self.clips * 1.5):]

    def collision_detect(self):
        '''
            #在更新完一张图片的所有物体之后执行本操作
            1. 通过线性回归判断一个物体是否靠近相机
            2. 通过方差确定一个物体是否在某一个位置不动，此操作可以判断物体是否朝相机的位置前进
            3. 通过坐标点的位置大致确定物体和相机的方向
        '''

        max_len = len(self.dataOfobject)
        id = 0
        obj_pack = {}
        while id < max_len:
            is_near = self.is_near(id)
            is_direction = self.is_direction(id)
            obj_pack[id] = ['near' if is_near else '', 'same' if is_direction else '']
            id += 1
        return obj_pack

    def reflect_to_background(self):
        background = draw_birdView()    #定义背景
        for id,obj_clips in enumerate(self.dataOfobject):
            onePath = []
            for x,y,z in obj_clips[7]:
                x,z = x.item(), z.item()
                # (xyz, T, c, name)
                # cp = (int((background.shape[1] / 2 + point[0][0] / 10)), int(background.shape[0] - point[0][2] / 10))
                cp = (int((background.shape[1] / 2 + x/10)), int(background.shape[0] - z/10))
                cv2.circle(background, cp, radius=3, color=colors(id, True), thickness=-1)
                onePath.append(cp)
            # print(onePath)
            if len(onePath) > 1:
                extend_points = self.predict_position(onePath,obj_clips[3])
                cv2.polylines(background, [np.array(onePath)], isClosed=False, color=colors(id, True), thickness=1)

                for p in extend_points:
                    cv2.line(background,onePath[-1],p, color=colors(id*100, True), thickness=1)
                # cv2.polylines(background, [np.array(extend_points)], isClosed=False, color=colors(id*2, True), thickness=1)
        cv2.line(background, (0, 1500), (1000, 1500), (100, 100, 100), 1)
        self.background = cv2.resize(background, (469*2, 720*2))
        cv2.imshow('dimension_map',self.background)



# obj = collision_detection()
# back_test = draw_birdView()
# pred = [(580, 1295), (571, 1337), (553, 1411), (542, 1456), (555, 1407)]
# point = [(585, 1276), (576, 1314), (567, 1359), (542, 1456), (542, 1456), (566, 1368)]
# while True:
#     # cv2.circle(back_test, p1, radius=2, color=colors(100, True), thickness=-1)
#     # cv2.circle(back_test, p2, radius=2, color=colors(10, True), thickness=-1)
#     # cv2.line(back_test, p2, p1, color=colors(10, True), thickness=1)
#
#     # cv2.polylines(back_test, [np.array(point)], isClosed=False, color=colors(1, True), thickness=1)
#     for p in pred:
#         cv2.line(back_test, point[-1], p, color=colors(1 * 100, True), thickness=1)
#
#     back_test = cv2.resize(back_test, (469, 720))
#     cv2.imshow('test',back_test)
#     cv2.waitKey(1)
