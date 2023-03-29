import socket
import sys
import time
import cv2
import numpy
# import threading
import multiprocessing
import numpy as np

def ReceiveVideo():
    print('This is HOST!')
    # IP地址'0.0.0.0'为等待客户端连接
    address = ('172.16.36.120', 8888)
    # 建立socket对象，参数意义见https://blog.csdn.net/rebelqsp/article/details/22109925
    # socket.AF_INET：服务器之间网络通信
    # socket.SOCK_STREAM：流式socket , for TCP
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # 将套接字绑定到地址, 在AF_INET下,以元组（host,port）的形式表示地址.
    s.bind(address)
    # 开始监听TCP传入连接。参数指定在拒绝连接之前，操作系统可以挂起的最大连接数量。该值至少为1，大部分应用程序设为5就可以了。
    s.listen(2)

    # 接受TCP连接并返回（conn,address）,其中conn是新的套接字对象，可以用来接收和发送数据。addr是连接客户端的地址。
    # 没有连接则等待有连接
    while True:
        conn, addr = s.accept()
        # t = threading.Thread(target=deal_data, args=(conn, addr))
        t = multiprocessing.Process(target=deal_data, args=(conn, addr))
        print('connect from:' + str(addr))
        try:
            t.start()
        except Exception as e:
            print(e)


def recvall(sock, count):
    buf = b''  # buf是一个byte类型
    while count:
        # 接受TCP套接字的数据。数据以字符串形式返回，count指定要接收的最大数据量.
        newbuf = sock.recv(count)
        if not newbuf: return None
        buf += newbuf
        count -= len(newbuf)
    return buf


def deal_data(conn, addr):
    print('Accept new connection from {0}'.format(addr))

    while True:
        start = time.time()  # 用于计算帧率信息
        try:
            length_RGB = recvall(conn, 16)  # 获得图片文件的长度,16代表获取长度
            length_D = recvall(conn, 16)  # 获得图片文件的长度,16代表获取长度
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
        depth = cv2.normalize(depth, None, 255, 0, cv2.NORM_MINMAX, cv2.CV_8UC1)
        depth = cv2.equalizeHist(depth)
        decimg_D = cv2.applyColorMap(255 - depth, cv2.COLORMAP_HOT)

        conn.send(str.encode('GUJIHAO'.ljust(7)))

        # 将帧率信息回传，主要目的是测试可以双向通信
        end = time.time()
        seconds = end - start
        fps = 1 / seconds
        cv2.putText(decimg_RGB, "fps: {:}".format(int(fps)), (2, decimg_RGB.shape[0] - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1)
        cv2.putText(decimg_D, "fps: {:}".format(int(fps)), (2, decimg_D.shape[0] - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1)

        cv2.imshow(f'RGB {str(addr[0])}:{str(addr[1])}', decimg_RGB)  # 显示图像
        cv2.imshow(f'D {str(addr[0])}:{str(addr[1])}', decimg_D)  # 显示图像

        print(int(seconds*1000), 'ms')
        if cv2.waitKey(1) == ord('q'):
            break



if __name__ == '__main__':
    ReceiveVideo()


