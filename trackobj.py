import cvzone

from collision import *
from yolov5.utils.general import (non_max_suppression, scale_coords, cv2,
                                  xyxy2xywh)
from yolov5.utils.plots import Annotator, colors
from yolov5.utils.torch_utils import time_sync



#参数初始化
dt, seen,t1,t2,t3,t4,t5,t6,t7  = [0.0, 0.0, 0.0, 0.0], 0,0,0,0,0,0,0,0
curr_frames, prev_frames = None, None
collision = collision_detection()
ts = time_sync()
pack = {}
#将测试保存为视频
# fourcc = cv2.VideoWriter_fourcc(*'mp4v')
# cap_fps = 20
# video = cv2.VideoWriter('result.mp4', fourcc, cap_fps, (800,500))
#开始


while True:

    im, im0, img_d ,s , img_depth = next_img(cap1,cap2)

    t1 = time_sync()
    im = torch.from_numpy(im).to(device)
    img_d = torch.from_numpy(img_d).to(device)

    im = im.half() if half else im.float()  # uint8 to fp16/32
    im /= 255.0
    t2 = time_sync()
    dt[0] += t2 - t1

    # Inference
    pred = model(im, augment=augment, visualize=visualize)
    t3 = time_sync()
    dt[1] += t3 - t2

    #nms
    pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
    dt[2] += time_sync() - t3

    # Process detections
    for det in pred:  # detections per image
        seen += 1
        curr_frames = im0
        s += f'{im.shape[2:]} '

        annotator = Annotator(im0, line_width=1, pil=not ascii,font_size=1)
        annotator_d = Annotator(img_depth, line_width=1, pil=not ascii,font_size=1)
        if cfg.STRONGSORT.ECC:  # camera motion compensation
            strongsort.tracker.camera_update(prev_frames, curr_frames)

        if det is not None and len(det):
            det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()

            # Print results
            for c in det[:, -1].unique():
                n = (det[:, -1] == c).sum()  # detections per class
                s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

            xywhs = xyxy2xywh(det[:, 0:4])
            confs = det[:, 4]
            clss = det[:, 5]
            xyxy = det[:, 0:4]

            # pass detections to strongsort
            t4 = time_sync()
            outputs = strongsort.update(xywhs.cpu(), confs.cpu(), clss.cpu(), im0)
            t5 = time_sync()
            dt[3] += t5 - t4

            if len(outputs) > 0:

                for j, (output, conf) in enumerate(zip(outputs, confs)):
                    notice = ''
                    bboxes = output[0:4]

                    id = output[4]
                    cls = output[5]

                    c = int(cls)  # integer class
                    id = int(id)  # integer id

                    dist = getObjectDepth(img_d,bboxes)
                    x = getX(dist, (bboxes[0] + bboxes[2]) / 2)
                    y = 0
                    center = ( int( (output[0]+output[2]) / 2 ),int( (output[1]+output[3])/2))

                    near = '' if pack.get(id) == None else pack[id][0]
                    dir = '' if pack.get(id) == None else pack[id][1]
                    if near and dir : notice = 'notice'
                    if dist < 7: notice = 'collision'
                    label =  f'{id} {names[c]} {conf * 100:.2f}% {dist/10:.2f}m {notice}'
                    annotator.box_label(bboxes, label, color=colors(c, True))
                    annotator_d.box_label(bboxes, label, color=colors(c, True))
                    cv2.circle(im0,center,radius=1,color=(255,0,255))
                    # 这里应该把outputs中所有的信息传递到物体碰撞检测中去

                    collision.update_object(id,dist.item()/10,center,names[c],time_sync()-ts,(x,y,dist))

        else:
            strongsort.increment_ages()
        # 这里应该更新每一个物体的寿命，当寿命为0的时候就将这个物体从物体的列表中给去掉

        t6 = time_sync()

        collision.update()
        pack = collision.collision_detect()


        t7  = time_sync()

        im0 = annotator.result()
        img_depth = annotator_d.result()

        im0 = letterbox(im0,(800,800),is_border=True)[0]

        # print(im0.shape[1],im0.shape[0])

        fps = 15 if t5 - t1 <= 0 else 1 / (t5 - t1)
        collision.update_clips(int(fps),1)
        cvzone.putTextRect(im0, f' YOLO:({t3 - t2:.3f}s), StrongSORT:({t5 - t4:.3f}s), collision:({t7-t6:.3f}s), FPS:({fps :.2f})', (0, 20),scale=1,thickness=1)
        # video.write(im0)

        collision.reflect_to_background()

        cv2.imshow('demo', im0)
        cv2.imshow('img_d',img_depth)

        cv2.waitKey(1)  # 1 millisecond
        # cc += 1
        # print(s)
        prev_frames = curr_frames
# video.release()

