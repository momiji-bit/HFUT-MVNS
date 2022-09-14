import matplotlib.pyplot as plt
import numpy as np
import time
from math import *

plt.ion() #开启interactive mode 成功的关键函数
plt.figure(1)
t = [0]
t_now = 0
m = [sin(t_now)]

for i in range(2000):
    plt.clf() #清空画布上的所有内容
    t_now = i*0.1
    t.append(t_now)#模拟数据增量流入，保存历史数据
    m.append(sin(t_now))#模拟数据增量流入，保存历史数据
    plt.plot(t,m,'-r')
    # plt.draw()#注意此函数需要调用
    # time.sleep(0.01)
    plt.pause(0.01)


