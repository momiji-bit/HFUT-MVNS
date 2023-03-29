import numpy as np
import librosa
import glob
import scipy
from IPython.display import Audio
import matplotlib.pyplot as pit

Trans_Audio = './coco80_CN_TT.m4a'

class HRTF:
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
        print(pitch, yaw, LR)

        return self.Hrir_Data[pitch][yaw][LR]


H = HRTF()

wv, wv_sr = librosa.load(Trans_Audio, sr=48000, mono=True)
Audio(wv, rate=48000)

L, _ = librosa.load(H.getHrir(pitch=0, yaw=0, LR='L'), sr=48000, mono=True)
R, _ = librosa.load(H.getHrir(pitch=0, yaw=0, LR='R'), sr=48000, mono=True)
conL = scipy.signal.convolve(wv, L)
conR = scipy.signal.convolve(wv, R)
conOut = np.vstack([conL, conR])
Audio(conOut, rate=48000)

L, _ = librosa.load(H.getHrir(pitch=0, yaw=45, LR='L'), sr=48000, mono=True)
R, _ = librosa.load(H.getHrir(pitch=0, yaw=45, LR='R'), sr=48000, mono=True)
conL = scipy.signal.convolve(wv, L)
conR = scipy.signal.convolve(wv, R)
conOut = np.vstack([conL, conR])
Audio(conOut, rate=48000)


L, _ = librosa.load(H.getHrir(pitch=-20, yaw=300, LR='L'), sr=48000, mono=True)
R, _ = librosa.load(H.getHrir(pitch=-20, yaw=300, LR='R'), sr=48000, mono=True)
conL = scipy.signal.convolve(wv, L)
conR = scipy.signal.convolve(wv, R)
conOut = np.vstack([conL, conR])
Audio(conOut, rate=48000)
