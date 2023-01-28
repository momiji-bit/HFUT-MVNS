from roadbound import RoadBoundGetter
import cv2 as cv
cap = cv.VideoCapture(r"4.mp4")
model=RoadBoundGetter(scale=0.3,density=10)#初始化一个RoadBoundGetter类，将原图宽高缩小到原来的0.3后再进行计算，道路边缘点提取的密度为10个点选一个点
if not cap.isOpened():
    print("Cannot open ")
    exit()
while True:
    ret, frame = cap.read()
    out=model(frame)#获取到道路边缘点集合
    show_img=frame.copy()
    for i in out:#画道路边缘点
        show_img=cv.circle(show_img,(int(i[0]),int(i[1])),radius=3,color=(0,0,255),thickness=-1,lineType=8)
    cv.imshow("result", cv.resize(show_img,dsize=(0,0),fx=0.5,fy=0.5))
    if cv.waitKey(1) == ord('q'):
        break
cap.release()
cv.waitKey(0)
cv.destroyAllWindows()
