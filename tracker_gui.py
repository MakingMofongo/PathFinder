import sys
import cv2
import yolo
import time
import path_finding as pf
import midas_single_image as midas
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QListWidget, QPushButton, QListWidgetItem
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import QTimer, Qt

def create_tracker():
    tracker = cv2.TrackerKCF_create()
    return tracker

def get_bounding_box(frame, objects, locations, index):
    xmin, ymin, xmax, ymax = [int(v) for v in locations[index]]
    return (xmin, ymin, xmax - xmin, ymax - ymin)

class VideoWidget(QWidget):
    def __init__(self):
        super().__init__()

        self.tracker = create_tracker()
        self.video = cv2.VideoCapture(0)
        self.frame = None
        self.objects = []
        self.locations = []
        self.bbox = None
        self.tracking = False
        self.tracked_objects = {}
        self.recent_objects = {}
        self.navigation_started = False

        self.video_label = QLabel(self)
        self.objects_list = QListWidget(self)
        self.recent_objects_list = QListWidget(self)
        self.back_button = QPushButton("Back to YOLO mode", self)
        self.navigate_button = QPushButton("Navigate", self)
        self.navigate_button.hide()

        video_layout = QVBoxLayout()
        video_layout.addWidget(self.video_label)
        video_layout.addWidget(self.objects_list)
        video_layout.addWidget(self.back_button)
        video_layout.addWidget(self.navigate_button)

        sidebar_layout = QVBoxLayout()
        sidebar_layout.addWidget(QLabel("Recently appeared objects:"))
        sidebar_layout.addWidget(self.recent_objects_list)

        main_layout = QHBoxLayout()
        main_layout.addLayout(video_layout)
        main_layout.addLayout(sidebar_layout)
        self.setLayout(main_layout)

        self.timer = QTimer()
        self.timer.timeout.connect(self.update_video)
        self.timer.start(30)

        self.recent_objects_timer = QTimer()
        self.recent_objects_timer.timeout.connect(self.update_recent_objects)
        self.recent_objects_timer.start(1000)

        self.recent_objects_list.itemClicked.connect(self.start_tracking)
        self.back_button.clicked.connect(self.switch_to_yolo_mode)
        self.navigate_button.clicked.connect(self.toggle_navigation)

    def update_video(self):
        ret, self.frame = self.video.read()
        if ret:
            if not self.tracking:
                result = yolo.inference(self.frame, Loop=False)
                self.frame = result[2]
                self.objects = result[0]
                self.locations = result[1]

                self.objects_list.clear()
                for obj in self.objects:
                    item = QListWidgetItem(obj)
                    self.objects_list.addItem(item)

                for obj_name, bbox in self.tracked_objects.items():
                    if obj_name in self.objects:
                        index = self.objects.index(obj_name)
                        self.bbox = get_bounding_box(self.frame, self.objects, self.locations, index)
                        self.tracker = create_tracker()
                        self.tracker.init(self.frame, self.bbox)
                        self.tracking = True
                        self.navigate_button.show()
                        break

            else:
                success, self.bbox = self.tracker.update(self.frame)

                if success:
                    x, y, w, h = [int(v) for v in self.bbox]
                    cv2.rectangle(self.frame, (x, y), (x + w, y + h), (0,255, 0), 2)
                    if self.navigation_started:
                        self.start_navigation()
                else:
                    self.switch_to_yolo_mode()

            height, width, channel = self.frame.shape
            bytes_per_line = 3 * width
            qimage = QImage(self.frame.data, width, height, bytes_per_line, QImage.Format_RGB888).rgbSwapped()
            self.video_label.setPixmap(QPixmap.fromImage(qimage))

    def update_recent_objects(self):
        current_time = time.time()
        for obj in self.objects:
            if obj not in self.recent_objects:
                self.recent_objects[obj] = current_time

        expired_objects = [obj for obj, timestamp in self.recent_objects.items() if current_time - timestamp > 30]
        for obj in expired_objects:
            del self.recent_objects[obj]

        self.recent_objects_list.clear()
        for obj in self.recent_objects:
            item = QListWidgetItem(obj)
            if obj in self.tracked_objects:
                item.setBackground(Qt.green)
            self.recent_objects_list.addItem(item)

    def start_tracking(self, item):
        obj_name = item.text()
        if obj_name in self.tracked_objects:
            del self.tracked_objects[obj_name]
            item.setBackground(Qt.white)
            self.navigate_button.hide()
        else:
            self.tracked_objects[obj_name] = None
            item.setBackground(Qt.green)
            if self.tracking:
                self.navigate_button.show()

    def switch_to_yolo_mode(self):
        self.tracking = False
        self.navigate_button.hide()

    def toggle_navigation(self):
        self.navigation_started = not self.navigation_started

    def start_navigation(self):
        if self.tracking and self.bbox:
            x, y, w, h = [int(v) for v in self.bbox]
            target_point = (x + w // 2, y + h)
            cv2.circle(self.frame, target_point, 5, (0, 0, 255), -1)
            # save self.frame
            save_path = './Midas/inputs/rgb/'
            cv2.imwrite(save_path + 'frame_gui.png', self.frame)
            # print size of self.frame
            print('Size of self frame',self.frame.shape)
            midas.run_midas(save_path)
            depth_map = midas._open_map()
            path_result = pf.path_to_point(target_point, depth_map)
            # if is not none
            if path_result is None:
                print('No path found')
            else:
                path = path_result[0]
                depth_map_with_path = path_result[1]

                for i in range(len(path) - 1):
                    cv2.line(self.frame, path[i], path[i + 1], (0, 0, 255), 2)
                cv2.imshow('Depth Map', depth_map_with_path)
                cv2.imshow('Navigation', self.frame)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = VideoWidget()
    window.show()
    sys.exit(app.exec_())
