import cv2

cap = cv2.VideoCapture('./depth.mp4')
time = 0
while cap.isOpened():
    _, frame = cap.read()
    if not _:
        break
    cv2.imwrite(f'./depth/{time}.png', frame)
    time+=1
    print(time)
print('done!')