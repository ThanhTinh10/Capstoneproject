import argparse
import time
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random
import serial

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized, TracedModel


def detect(source, weights, device, img_size, iou_thres, conf_thres, arduino_port):
# def detect(source, weights, device, img_size, iou_thres, conf_thres):


    webcam = source.isnumeric()



    # Initialize
    set_logging()
    device = select_device(device)
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(img_size, s=stride)  # check img_size


    if half:
        model.half()  # to FP16


    # Set Dataloader
    if webcam:
        view_img = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride)



    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
    old_img_w = old_img_h = imgsz
    old_img_b = 1

    # t0 = time.time()
    t0 = time.perf_counter()

    # Serial communication with Arduino
    # arduino = serial.Serial(arduino_port, 9600)
    arduino = serial.Serial(arduino_port, 9600)
    count_good1 = 0
    count_good2 = 0
    count_bad = 0
    lemon_detected = False
    # start_time = 0  # Thời điểm bắt đầu khi quả chanh vào khung hình

    for path, img, im0s, vid_cap in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Warmup
        if device.type != 'cpu' and (old_img_b != img.shape[0] or old_img_h != img.shape[2] or old_img_w != img.shape[3]):
            old_img_b = img.shape[0]
            old_img_h = img.shape[2]
            old_img_w = img.shape[3]


        # Inference
        t1 = time_synchronized()
        with torch.no_grad():   # Calculating gradients would cause a GPU memory leak
            pred = model(img)[0]
        t2 = time_synchronized()

        # Apply NMS
        pred = non_max_suppression(pred, conf_thres, iou_thres)
        t3 = time_synchronized()

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            if webcam:  # batch_size >= 1
                p, s, im0, frame = path[i], '%g: ' % i, im0s[i].copy(), dataset.count

            p = Path(p)  # to Path

            # save_path = str(save_dir / p.name)  # img.jpg
            # txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # img.txt

            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string
                for *xyxy, conf, cls in reversed(det):
                    # if save_txt:  # Write to file
                    #     xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                    #     line = (cls, *xywh, conf) if opt.save_conf else (cls, *xywh)  # label format
                    #     with open(txt_path + '.txt', 'a') as f:
                    #         f.write(('%g ' * len(line)).rstrip() % line + '\n')

                    # if save_img or view_img:  # Add bbox to image
                    label = f'{names[int(cls)]}'
                    # label = f'{names[int(cls)]} {conf:.2f}'
                    plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=1)

                    if label == 'Good1':
                        count_good1 += 1
                    if label == 'Good2':
                        count_good2 += 1
                    if label == 'Bad':
                        count_bad += 1
                    # lemon_detected = True

                    if count_good1 > 0 or count_good2 > 0 or count_bad > 0 :
                        lemon_detected = True
                        # start_time = time.time()

                    # if label in ['Good1', 'Good2', 'Bad']:
                    #     lemon_detected = True
                    #     # start_time = time.time()

            if lemon_detected and count_good1 > 0 and count_good2 == 0 and count_bad == 0:
                arduino.write(b'1')
                time.sleep(1)
                lemon_detected = False

            if lemon_detected and count_good1 == 0 and count_good2 > 0 and count_bad == 0:
                arduino.write(b'2')
                time.sleep(1)
                lemon_detected = False

            if lemon_detected and count_good1 > 0 and count_good2 > 0 and count_bad == 0:
                arduino.write(b'2')
                time.sleep(1)
                lemon_detected = False

            if lemon_detected and count_good1 > 0 and count_good2 > 0 and count_bad > 0:
                arduino.write(b'3')
                time.sleep(1)
                lemon_detected = False

            if lemon_detected and count_good1 == 0 and count_good2 == 0 and count_bad > 0:
                arduino.write(b'3')
                time.sleep(1)
                lemon_detected = False

            if lemon_detected and count_good1 > 0 and count_good2 == 0 and count_bad > 0:
                arduino.write(b'3')
                time.sleep(1)
                lemon_detected = False

            if lemon_detected and count_good1 == 0 and count_good2 > 0 and count_bad > 0:
                arduino.write(b'3')
                time.sleep(1)
                lemon_detected = False

            count_good1 = 0
            count_good2 = 0
            count_bad = 0










        # cv2.imshow(str(p),im0)

    arduino.close()



    print(f'Done. ({time.time() - t0:.3f}s)')


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(device)

    with torch.no_grad():
        # detect("1", "best.pt", device, img_size=640, iou_thres=0.45, conf_thres=0.6, arduino_port='COM9')
        detect("0", "best.pt", device, img_size=640, iou_thres=0.45, conf_thres=0.6)


