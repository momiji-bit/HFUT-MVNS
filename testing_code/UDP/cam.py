import socket
import time

import cv2
import numpy as np

addr = ('localhost', 8888)
s = socket.socket(socket.AF_INET6, socket.SOCK_DGRAM)

cap = cv2.VideoCapture('/Users/gujihao/Desktop/2023-1-21 19.53拍摄的影片.mov')

while cap.isOpened():
    _, img = cap.read()
    img = cv2.resize(img, (640, 360))
    _, img_jpeg = cv2.imencode('.jpg', img, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
    img_array = np.array(img_jpeg)  # UDP 传输最大字节数 9217
    img_byte = img_array.tobytes()
    s.sendto(str.encode(str(len(img_byte)).ljust(8)) + img_byte, addr)
