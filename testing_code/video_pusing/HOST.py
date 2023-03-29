# 相机
# 仅测试用 调用笔记本摄像头 深度摄像头在本代码中就是RGB摄像头
import socket
import cv2
import time
import numpy as np


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
    address = ('127.0.0.1', 8080)
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(address)
    s.listen(1)  # 设置可同时连接的客户端个数
    cap = cv2.VideoCapture(0)  # 这里设置摄像头编号0，1，2
    while True:
        conn, addr = s.accept()  # 网页套接字
        device_code = recvall(conn, 20).decode('utf-8')
        print(device_code)

        _, rgb = cap.read()  # 读相机画面
        # 在这里对图像进行拼接
        depth = rgb  # 这里没有深度摄像头，咱用RGB数据充当DEPTH数据
        img = np.hstack((rgb, depth))
        _, img_code = cv2.imencode('.jpg', img, [int(cv2.IMWRITE_JPEG_QUALITY), 95])  # 编码为jpg，压缩率95太低吃CPU性能
        img_numpy = np.array(img_code)  # 转numpy数组格式
        img_byte = img_numpy.tobytes()  # 转字节

        conn.send(str.encode(str(len(img_byte)).ljust(8)) + img_byte)  # TCP发送出去


