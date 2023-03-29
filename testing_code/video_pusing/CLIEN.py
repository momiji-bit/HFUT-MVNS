# 网页
import socket
import sys
import cv2
import numpy as np
import time


def recvall(sock, count):
    buf = b''  # buf是一个byte类型
    while count:
        # 接受TCP套接字的数据。数据以字符串形式返回，count指定要接收的最大数据量.
        newbuf = sock.recv(count)
        if not newbuf: return None
        buf += newbuf
        count -= len(newbuf)
    return buf


if __name__ == '__main__':
    while True:
        t0 = time.time()
        # TCP连接
        address = ('127.0.0.1', 8080)
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.connect(address)
        except socket.error as msg:
            print(msg)
            sys.exit(1)
        sock.send(str.encode('01abcd').ljust(20))  # 发送设备码

        # 获取相机数据
        try:
            length_RGB = recvall(sock, 8)  # 获得图片文件的长度,8代表获取长度
            stringData_RGB = recvall(sock, int(length_RGB))  # 根据获得的文件长度，获取图片文件
        except Exception as e:
            print(e)
            break
        # 数据解码
        data_RGB = np.frombuffer(stringData_RGB, np.uint8)  # 将获取到的字符流数据转换成1维数组
        decimg_RGB = cv2.imdecode(data_RGB, cv2.IMREAD_COLOR)  # 将数组解码成图像

        cv2.putText(decimg_RGB, "fps: {:}".format(int(1 / (time.time() - t0))), (2, decimg_RGB.shape[0] - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1)
        cv2.imshow('Demo', decimg_RGB)  # 显示图像
        if cv2.waitKey(1) == ord('q'):
            break
