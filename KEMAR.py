import numpy as np
import librosa
import glob

import scipy.signal
from scipy import signal
from IPython.display import Audio
import matplotlib.pyplot as pit


#
# Trans_Audio = './Summer_Ghost.wav'
#
# HRIR_D40_Dir = './HRIR_Data/elev-40/*.wav'
# HRIR_D30_Dir = './HRIR_Data/elev-30/*.wav'
# HRIR_D20_Dir = './HRIR_Data/elev-20/*.wav'
# HRIR_D10_Dir = './HRIR_Data/elev-10/*.wav'
# HRIR_DU0_Dir = './HRIR_Data/elev0/*.wav'
# HRIR_U10_Dir = './HRIR_Data/elev10/*.wav'
# HRIR_U20_Dir = './HRIR_Data/elev20/*.wav'
# HRIR_U30_Dir = './HRIR_Data/elev30/*.wav'
# HRIR_U40_Dir = './HRIR_Data/elev40/*.wav'
# HRIR_U50_Dir = './HRIR_Data/elev50/*.wav'
# HRIR_U60_Dir = './HRIR_Data/elev60/*.wav'
# HRIR_U70_Dir = './HRIR_Data/elev70/*.wav'
# HRIR_U80_Dir = './HRIR_Data/elev80/*.wav'
# HRIR_U90_Dir = './HRIR_Data/elev90/*.wav'
#
# HRIR_D40 = glob.glob(HRIR_D40_Dir)
# HRIR_D30 = glob.glob(HRIR_D30_Dir)
# HRIR_D20 = glob.glob(HRIR_D20_Dir)
# HRIR_D10 = glob.glob(HRIR_D10_Dir)
# HRIR_DU0 = glob.glob(HRIR_DU0_Dir)
# HRIR_U10 = glob.glob(HRIR_U10_Dir)
# HRIR_U20 = glob.glob(HRIR_U20_Dir)
# HRIR_U30 = glob.glob(HRIR_U30_Dir)
# HRIR_U40 = glob.glob(HRIR_U40_Dir)
# HRIR_U50 = glob.glob(HRIR_U50_Dir)
# HRIR_U60 = glob.glob(HRIR_U60_Dir)
# HRIR_U70 = glob.glob(HRIR_U70_Dir)
# HRIR_U80 = glob.glob(HRIR_U80_Dir)
# HRIR_U90 = glob.glob(HRIR_U90_Dir)
#
# wv, wv_sr = librosa.load(Trans_Audio, sr=48000, mono=True)


class HRIR:
    def __init__(self, path='./orientation/'):
        self.Hrir_List = glob.glob(path + '*.wav')
        self.Hrir_Data = dict()
        for data in self.Hrir_List:
            name = data.split('/')[-1].split('.')[0]
            LR = name[0]
            pitch = int(name[1:].split('e')[0])
            yaw = int(name[1:].split('e')[-1][:-1])
            if pitch not in self.Hrir_Data:
                self.Hrir_Data[pitch] = dict()
            if yaw not in self.Hrir_Data[pitch]:
                self.Hrir_Data[pitch][yaw] = dict()
            self.Hrir_Data[pitch][yaw][LR] = data

    def getHrir(self, pitch, yaw, LR):
        """
        输入俯仰角(-40~90)和偏航角(0~360)，返回左右耳的Hrir数据
        :param pitch: 俯仰角(-40~90)
        :param yaw: 偏航角(0~360)
        :param LR: 左右耳
        :return: Hrir路径
        """
        if pitch not in self.Hrir_Data:
            min = 180
            pitch += 0.5
            near_pitch = pitch
            for i in self.Hrir_Data:
                if abs(pitch-i) <= min:
                    min = abs(pitch-i)
                    near_pitch = i
            pitch = near_pitch
        if yaw not in self.Hrir_Data[pitch]:
            min = 360
            yaw += 0.5
            near_yaw = yaw
            for i in self.Hrir_Data[pitch]:
                if abs(yaw - i) <= min:
                    min = abs(yaw - i)
                    near_yaw = i
            yaw = near_yaw

        return self.Hrir_Data[pitch][yaw][LR]


H = HRIR()


L, _ = librosa.load(H.getHrir(pitch=34, yaw=5, LR='L'), sr=48000, mono=True)

scipy.signal.convolve()