from oak_Cam import *
from yolov8_tracking.run_yolov8_mvns import *

retina_masks = True  # whether to plot masks in native resolution
bs = 1
curr_frames, prev_frames = [None] * bs, [None] * bs

if __name__ == '__main__':
    oakCam = OakCam()
    msg = oakCam.get_msg()

    i = 0
    while True:
        t0 = time.time()
        RGB, D, IMU, latencyRGB, latencyD = next(msg)  # 获取OAK相机RGB D 数据
        p, img, im0, im, masks, proto = run_img(RGB)

        # Process predictionsf len(det) else '(no detections), '}{dt[1].dt * 1E3:.1f}ms")
        # Process detections
        det = p[0]
        s = 'oak: '
        s += '%gx%g ' % im.shape[2:]  # print string
        im0 = im0[0]
        annotator = Annotator(im0, line_width=2, example=str(names))
        if hasattr(tracker_list[i], 'tracker') and hasattr(tracker_list[i].tracker, 'camera_update'):
            if prev_frames[i] is not None and curr_frames[i] is not None:  # camera motion compensation
                tracker_list[i].tracker.camera_update(prev_frames[i], curr_frames[i])

        if det is not None and len(det):
            if is_seg:
                shape = im0.shape
                # scale bbox first the crop masks
                if retina_masks:
                    det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], shape).round()  # rescale boxes to im0 size
                    masks.append(process_mask_native(proto[i], det[:, 6:], det[:, :4], im0.shape[:2]))  # HWC
                else:
                    masks.append(process_mask(proto[i], det[:, 6:], det[:, :4], im.shape[2:], upsample=True))  # HWC
                    det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], shape).round()  # rescale boxes to im0 size
            else:
                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()  # rescale boxes to im0 size

            # Print results
            for c in det[:, 5].unique():
                n = (det[:, 5] == c).sum()  # detections per class
                s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

            # pass detections to strongsort
            outputs[i] = tracker_list[i].update(det.cpu(), im0)

            # draw boxes for visualization
            if len(outputs[i]) > 0:

                if is_seg:
                    # Mask plotting
                    annotator.masks(
                        masks[i],
                        colors=[colors(x, True) for x in det[:, 5]],
                        im_gpu=torch.as_tensor(im0, dtype=torch.float16).to(device).permute(2, 0, 1).flip(
                            0).contiguous() /
                               255 if retina_masks else im[i]
                    )

                for j, (output) in enumerate(outputs[i]):
                    bbox = output[0:4]
                    id = output[4]
                    cls = output[5]
                    conf = output[6]

                    c = int(cls)  # integer class
                    id = int(id)  # integer id
                    label = f'{id} {names[c]} {conf:.2f}'
                    color = colors(c, True)
                    annotator.box_label(bbox, label, color=color)
        for m in masks[0]:
            s = m.cpu().numpy()
            array1 = s * 255 / s.max()  # normalize，将图像数据扩展到[0,255]
            mat = np.uint8(array1)  # float32-->uint8
            cv2.imshow("img", mat)
            cv2.waitKey(1)

        # Stream results
        im0 = annotator.result()
        cv2.imshow('Demo', im0)
        if cv2.waitKey(1) == ord('q'):  # 1 millisecond
            exit()
        print(1/(time.time() - t0))
        prev_frames[i] = curr_frames[i]
