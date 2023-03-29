import socket
import numpy as np
import cv2

addr = ('localhost', 8888)
s = socket.socket(socket.AF_INET6, socket.SOCK_DGRAM)
s.bind(addr)

while True:
    data, addr = s.recvfrom(900000)
    data = data[8:8+int(data[:8])]
    img = np.frombuffer(data, np.uint8)  # 将获取到的字符流数据转换成1维数组
    img = cv2.imdecode(img, cv2.IMREAD_COLOR)  # 将数组解码成图像
    cv2.imshow('demo',img)
    if cv2.waitKey(1) == ord('q'):
        break

