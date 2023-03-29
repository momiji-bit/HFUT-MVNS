import yolov5.run_yolov5_mvns as yolov5
from yolov5.run_yolov5_mvns import *

from oak_Cam import *

# yolov5.run()


if __name__ == '__main__':
    oakCam = OakCam()
    msg = oakCam.get_msg()
    while True:
        RGB, D, IMU = next(msg)  # 获取OAK相机RGB D 数据
        pred, img, im0, im = yolov5.run_img(RGB)

        # Process predictionsf len(det) else '(no detections), '}{dt[1].dt * 1E3:.1f}ms")
        for i, det in enumerate(pred):  # per image
            im0 = im0[i].copy()
            s = 'cam:'
            s += f'{i}: '

            s += '%gx%g ' % im.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            annotator = Annotator(im0, line_width=3, example=str(names))
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, 5].unique():
                    n = (det[:, 5] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    c = int(cls)  # integer class
                    label = f'{names[c]} {conf:.2f}'
                    annotator.box_label(xyxy, label, color=colors(c, True))

            # Stream results
            im0 = annotator.result()
            cv2.imshow('cam', im0)
            cv2.waitKey(1)  # 1 millisecond
