import socket
import sys
import time
import cv2
import numpy
# import threading
import multiprocessing
import numpy as np
import run
import torch


def recvall(sock, count):
    buf = b''  # buf是一个byte类型
    while count:
        # 接受TCP套接字的数据。数据以字符串形式返回，count指定要接收的最大数据量.
        newbuf = sock.recv(count)
        if not newbuf: return None
        buf += newbuf
        count -= len(newbuf)
    return buf


def s():
    print('This is HOST!')
    address = ('192.168.31.54', 8080)
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(address)
    s.listen(2)
    while True:
        conn, addr = s.accept()
        while True:
            try:
                length_RGB = recvall(conn, 16)  # 获得图片文件的长度,16代表获取长度
                length_D = recvall(conn, 16)  # 获得图片文件的长度,16代表获取长度`
                stringData_RGB = recvall(conn, int(length_RGB))  # 根据获得的文件长度，获取图片文件
                stringData_D = recvall(conn, int(length_D))  # 根据获得的文件长度，获取图片文件
            except Exception as e:
                print(e)
                break
            data_RGB = numpy.frombuffer(stringData_RGB, numpy.uint8)  # 将获取到的字符流数据转换成1维数组
            data_D = numpy.frombuffer(stringData_D, numpy.uint8)  # 将获取到的字符流数据转换成1维数组
            decimg_RGB = cv2.imdecode(data_RGB, cv2.IMREAD_COLOR)  # 将数组解码成图像
            decimg_D = cv2.imdecode(data_D, cv2.IMREAD_COLOR)  # 将数组解码成图像
            depth = cv2.cvtColor(decimg_D, cv2.COLOR_BGR2GRAY)

            run.go(decimg_RGB, depth)
            cv2.waitKey(1)

if __name__ == '__main__':
    s()
