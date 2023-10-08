import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QLabel, QLineEdit, QPushButton, QTabWidget, QVBoxLayout
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QImage
from PyQt5.QtCore import QTimer
import cv2
import threading
import subprocess
from PyQt5.QtGui import QPainter, QPen
from PyQt5.QtWidgets import QGraphicsView, QGraphicsScene, QMainWindow, QApplication, QVBoxLayout, QWidget


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



class LoginWindow(QWidget):
    def __init__(self, main_app):
        super().__init__()

        self.main_app = main_app

        self.setWindowTitle("Login Screen")
        self.setGeometry(100, 100, 800, 600)

        self.username_label = QLabel("Username:", self)
        self.password_label = QLabel("Password:", self)
        self.username_input = QLineEdit(self)
        self.password_input = QLineEdit(self)

        self.username_label.move(200, 280)
        self.username_input.move(270, 280)
        self.password_label.move(200, 320)
        self.password_input.move(270, 320)

        self.username_input.setFixedWidth(200)
        self.password_input.setFixedWidth(200)

        self.username_labeles = QLabel("ĐỒ ÁN TỐT NGHIỆP - ME4327", self)
        self.username_labeles.move(170, 230)
        font = self.username_labeles.font()
        font.setPointSize(18)
        font.setBold(True)
        self.username_labeles.setFont(font)
        self.login_button = QPushButton("Log in ", self)
        self.login_button.move(320, 360)
        self.login_button.clicked.connect(self.check_login)

    def check_login(self):
        username = self.username_input.text()
        password = self.password_input.text()

        if username == "admin" and password == "123":
            self.close()
            self.main_app.show()
        else:
            self.username_input.clear()
            self.password_input.clear()
            self.username_input.setFocus()


class MenuTab(QWidget):
    def __init__(self, login_window):
        super().__init__()

        self.login_window = login_window
        self.layout = QVBoxLayout()
        self.image_label = QLabel(self)
        self.layout.addWidget(self.image_label)

        self.text_labels = QLabel("ĐẠI HỌC QUỐC GIA TP. HỒ CHÍ MINH", self)
        self.text_labels.move(370, 110)
        font = self.text_labels.font()
        font.setPointSize(16)
        font.setBold(True)
        self.text_labels.setFont(font)
        self.text_labels.setStyleSheet("color: blue;")

        self.text_label = QLabel("TRƯỜNG ĐẠI HỌC BÁCH KHOA", self)
        self.text_label.move(400, 140)
        font = self.text_label.font()
        font.setPointSize(16)
        font.setBold(True)
        self.text_label.setFont(font)
        self.text_label.setStyleSheet("color: blue;")

        self.text_labels = QLabel("KHOA CƠ KHÍ - BỘ MÔN CƠ ĐIỆN TỬ", self)
        self.text_labels.move(380, 170)
        font = self.text_labels.font()
        font.setPointSize(16)
        font.setBold(True)
        self.text_labels.setFont(font)
        self.text_labels.setStyleSheet("color: blue;")

        self.text_labelss = QLabel("Đề tài", self)
        self.text_labelss.move(540, 245)
        font = self.text_labelss.font()
        font.setPointSize(16)
        font.setBold(True)
        self.text_labelss.setFont(font)

        self.text_labels = QLabel("ĐỒ ÁN TỐT NGHIỆP-ME4327", self)
        self.text_labels.move(425, 210)
        font = self.text_labels.font()
        font.setPointSize(16)
        font.setBold(True)
        self.text_labels.setFont(font)
        self.text_labels.setStyleSheet("color: blue;")

        self.text_labels = QLabel("THIẾT KẾ HỆ THỐNG PHÂN LOẠI CHẤT", self)
        self.text_labels.move(370, 280)
        font = self.text_labels.font()
        font.setPointSize(16)
        font.setBold(True)
        self.text_labels.setFont(font)
        self.text_labels.setStyleSheet("color: blue;")

        self.text_labels = QLabel("LƯỢNG CHANH ỨNG DỤNG AI VISION", self)
        self.text_labels.move(370, 310)
        font = self.text_labels.font()
        font.setPointSize(16)
        font.setBold(True)
        self.text_labels.setFont(font)
        self.text_labels.setStyleSheet("color: blue;")

        self.text_labels = QLabel("Sinh viên thực hiện:", self)
        self.text_labels.move(400, 350)
        font = self.text_labels.font()
        font.setPointSize(16)
        font.setBold(True)
        self.text_labels.setFont(font)

        self.text_labels = QLabel("Khằm Thanh Tình - 1915531", self)
        self.text_labels.move(480, 380)
        font = self.text_labels.font()
        font.setPointSize(16)
        font.setBold(True)
        self.text_labels.setFont(font)
        self.text_labels.setStyleSheet("color: blue;")

        self.text_labels = QLabel("Giảng viên hướng dẫn: ", self)
        self.text_labels.move(400, 410)
        font = self.text_labels.font()
        font.setPointSize(16)
        font.setBold(True)
        self.text_labels.setFont(font)

        self.text_labels = QLabel("PGS.TS.Lê Đức Hạnh ", self)
        self.text_labels.move(480, 440)
        font = self.text_labels.font()
        font.setPointSize(16)
        font.setBold(True)
        self.text_labels.setFont(font)
        self.text_labels.setStyleSheet("color: blue;")



    def load_image(self, image_path):
        pixmap = QPixmap(image_path)
        scaled_pixmap = pixmap.scaled(300, 400, aspectRatioMode=Qt.KeepAspectRatio)
        self.image_label.setPixmap(scaled_pixmap)
        self.image_label.move(50, 150)
        self.image_label.setScaledContents(True)


class CameraTab(QWidget):


    def __init__(self):
        super().__init__()

        self.capture = None
        self.is_running = False
        self.camera_thread = None


        self.image_labels = QLabel(self)
        self.image_labels.setGeometry(5, 290, 300, 300)
        self.image_labels.setPixmap(QPixmap("logitech.jpg"))

        self.text_labels = QLabel("Product Specifications ", self)
        self.text_labels.move(48, 130)
        font = self.text_labels.font()
        font.setPointSize(13)
        font.setBold(True)
        self.text_labels.setFont(font)


        self.text_labels = QLabel(" Brand______Logitech ", self)
        self.text_labels.move(48, 155)
        font = self.text_labels.font()
        font.setPointSize(13)
        font.setBold(True)
        self.text_labels.setFont(font)

        self.text_labels = QLabel("Colour______Black ", self)
        self.text_labels.move(50, 180)
        font = self.text_labels.font()
        font.setPointSize(13)
        font.setBold(True)
        self.text_labels.setFont(font)

        self.text_labels = QLabel("Model______C615 ", self)
        self.text_labels.move(55, 205)
        font = self.text_labels.font()
        font.setPointSize(13)
        font.setBold(True)
        self.text_labels.setFont(font)





        self.start_button = QPushButton("Check Camera", self)
        self.start_button.setFixedWidth(200)
        self.start_button.move(45, 60 )
        self.start_button.clicked.connect(self.start_camera)
        # self.layout.addWidget(self.start_button)



        self.run_button = QPushButton("Run Model", self)
        self.run_button.setFixedWidth(200)
        self.run_button.move(45, 95)
        self.run_button.clicked.connect(self.toggle_A)

        # show camera
        self.camera_label = QLabel(self)
        self.camera_label.setGeometry(310, 30, 500, 500)
        self.camera_labeles = QLabel(self)
        self.camera_labeles.setGeometry(10, 200, 270, 270)

        # connect state
        self.status_label = QLabel("", self)
        self.status_label.setGeometry(10, 300, 400, 400)
        self.camera_running = False


    def load_images(self, image_paths):
        pixmap = QPixmap(image_paths)
        scaled_pixmap = pixmap.scaled(300, 400, aspectRatioMode=Qt.KeepAspectRatio)
        self.image_labels.setPixmap(scaled_pixmap)
        self.image_labels.move(20, 250)
        self.image_labels.setScaledContents(True)


    def update_camera(self):
        if self.camera_running:
            ret, frame = self.capture.read()
            if ret:
                # show camera at camera_label
                height, width, channel = frame.shape
                bytes_per_line = 3 * width
                q_img = QImage(frame.data, width, height, bytes_per_line, QImage.Format_RGB888)
                pixmap = QPixmap.fromImage(q_img)
                self.camera_label.setPixmap(pixmap)
                self.camera_label.setScaledContents(True)

            QTimer.singleShot(30, self.update_camera)



    def start_camera(self):
        if not self.camera_running:
            self.capture = cv2.VideoCapture(1, cv2.CAP_DSHOW)
            if self.capture.isOpened():
                self.camera_running = True
                self.start_button.setText("Stop Camera")
                self.update_camera()
                self.layout = QVBoxLayout()
                self.status_label.setText("The camera has connected successfully")
                font = self.status_label.font()
                font.setPointSize(11)
                font.setBold(True)
                self.status_label.setFont(font)
                self.status_label.move(10, 100)
                self.camera_label.clear()



            else:


                self.camera_label.setText("Can't connect to camera. Please check and reconnect")
                font = self.camera_label.font()
                font.setPointSize(11)
                font.setBold(True)
                self.camera_label.setFont(font)
                self.camera_label.move(10, 100)
                self.status_label.clear()


        else:
            self.camera_running = False
            self.start_button.setText("Check Camera")
            self.camera_label.clear()
            self.capture.release()
            self.status_label.clear()

    def toggle_A(self):
        if not self.is_running:
            self.camera_thread = threading.Thread(target=self.run_A)
            self.camera_thread.start()
            self.is_running = True
            self.run_button.setText("Stop Model")


        else:

            self.is_running = False
            self.run_button.setText("Run Model")
            self.stop_camera()
            self.camera_thread.join()

    def run_A(self):

        def detect(source, weights, device, img_size, iou_thres, conf_thres):

            webcam = source.isnumeric()
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
            t0 = time.perf_counter()
            arduino = serial.Serial(arduino_port, 9600)
            count_good1 = 0
            count_good2 = 0
            count_bad = 0
            lemon_detected = False

            for path, img, im0s, vid_cap in dataset:
                img = torch.from_numpy(img).to(device)
                img = img.half() if half else img.float()  # uint8 to fp16/32
                img /= 255.0  # 0 - 255 to 0.0 - 1.0
                if img.ndimension() == 3:
                    img = img.unsqueeze(0)

                # Warmup
                if device.type != 'cpu' and (
                        old_img_b != img.shape[0] or old_img_h != img.shape[2] or old_img_w != img.shape[3]):
                    old_img_b = img.shape[0]
                    old_img_h = img.shape[2]
                    old_img_w = img.shape[3]

                # Inference
                t1 = time_synchronized()
                with torch.no_grad():  # Calculating gradients would cause a GPU memory leak
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

                            if count_good1 > 0 or count_good2 > 0 or count_bad > 0:
                                lemon_detected = True




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

                cv2.imshow(str(p), im0)
                self.status_label.clear()
                self.camera_labeles.setText("Model is ready")
                self.camera_labeles.move(40, 165)
                font = self.camera_labeles.font()
                font.setPointSize(11)
                font.setBold(True)
                self.camera_labeles.setFont(font)



            arduino.close()
            print(f'Done. ({time.time() - t0:.3f}s)')
        self.status_label.setText("Model is running....")
        self.status_label.move(40, 100)
        font = self.status_label.font()
        font.setPointSize(11)
        font.setBold(True)
        self.status_label.setFont(font)

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(device)

        with torch.no_grad():
            detect("1", "best.pt", device, img_size=640, iou_thres=0.45, conf_thres=0.6)



class MainApp(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("QMainWindow")
        self.setGeometry(100, 100, 800, 600)

        self.tabs = QTabWidget()
        self.menu_tab = MenuTab(self)
        self.camera_tab = CameraTab()

        self.tabs.addTab(self.menu_tab, "Home")
        self.tabs.addTab(self.camera_tab, "Camera")

        self.setCentralWidget(self.tabs)
    def show_menu_tab(self):
        self.tabs.setCurrentWidget(self.menu_tab)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    main_app = MainApp()
    login_window = LoginWindow(main_app)
    image_path = r'C:\Users\Administrator\PycharmProjects\pythonProject1\venv\yolov7\logo.jpg'
    main_app.menu_tab.load_image(image_path)
    login_window.show()
    sys.exit(app.exec_())
