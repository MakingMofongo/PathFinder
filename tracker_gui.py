import sys
import cv2
import yolo
import time
import path_finding as pf
import midas_single_image as midas
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QListWidget, QPushButton, QListWidgetItem, QPlainTextEdit
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import QTimer, Qt
import blip
import deepface_recognizer as deepface
from ElevenLabs.eleven_labs import play_audio
import numpy as np


def create_tracker():
    # Create tracker object
    # tracker = cv2.TrackerCSRT_create() # slow, accurate, bad track loss detection
    tracker = cv2.TrackerKCF_create() # fast, less accurate, good track loss detection

    return tracker

def get_bounding_box(frame, objects, locations, index):
    xmin, ymin, xmax, ymax = [int(v) for v in locations[index]]
    return (xmin, ymin, xmax - xmin, ymax - ymin)

class VideoWidget(QWidget):
    def __init__(self):
        
        super().__init__()
        self.yolo_model = yolo.load_yolo()
        self.blip_processor, self.blip_model = blip.setup_model()
        self.tracker = create_tracker()
        self.video = cv2.VideoCapture(0)
        self.frame = None
        self.raw_frame = None
        self.objects = []
        self.locations = []
        self.bbox = None
        self.tracking = False
        self.tracked_objects = {}
        self.recent_objects = {}
        self.navigation_started = False
        self.model, self.transform, self.net_w, self.net_h, self.device = midas.init()

        self.video_label = QLabel(self)
        self.objects_list = QListWidget(self)
        self.recent_objects_list = QListWidget(self)
        self.back_button = QPushButton("Back to YOLO mode", self)
        self.navigate_button = QPushButton("Navigate", self)
        self.navigate_button.hide()
        self.caption_textbox = QPlainTextEdit(self)
        self.caption_textbox.setReadOnly(True)
        self.describe_button = QPushButton("Describe environment", self)
        self.facerecognize_button = QPushButton("Recognize face", self) 
        self.faceinfer_button = QPushButton("Infer face", self)
        self.face_textbox = QPlainTextEdit(self)
        self.face_textbox.setReadOnly(True)

        video_layout = QVBoxLayout()
        video_layout.addWidget(self.video_label)
        video_layout.addWidget(self.objects_list)
        video_layout.addWidget(self.back_button)
        video_layout.addWidget(self.navigate_button)
        video_layout.addWidget(self.describe_button)
        video_layout.addWidget(self.caption_textbox)
        video_layout.addWidget(self.facerecognize_button)
        video_layout.addWidget(self.faceinfer_button)
        video_layout.addWidget(self.face_textbox)

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
        self.recent_objects_timer.start(3000)

        self.recent_objects_list.itemClicked.connect(self.start_tracking)
        self.back_button.clicked.connect(self.switch_to_yolo_mode)
        self.navigate_button.clicked.connect(self.toggle_navigation)
        self.describe_button.clicked.connect(self.describe_environment)
        self.facerecognize_button.clicked.connect(self.recognize_face)
        self.faceinfer_button.clicked.connect(self.infer_face)

    def update_video(self):
        ret, self.frame = self.video.read()
        self.raw_frame = self.frame.copy()
        if ret:
            if not self.tracking:
                result = yolo.inference(self.frame, model=self.yolo_model)
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
        if self.navigation_started:
            self.navigate_button.setText("Navigating...")
            # set button color
            self.navigate_button.setStyleSheet("background-color: green")

        else:
            self.navigate_button.setText("Navigate")
            # set button color
            self.navigate_button.setStyleSheet("background-color: white")
    
    def recognize_face(self):
        if self.frame is not None:
            # Convert the frame to an RGB image
            image = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)

            # Generate the caption using the Blip model
            name, x, y, w, h = deepface.recognize_face(image)

            # Display the caption in the text box
            self.face_textbox.setPlainText(name)
            play_audio(name)
    
    def infer_face(self):
        if self.frame is not None:
            # Convert the frame to an RGB image
            image = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)

            age,gender,race,emotion = deepface.infer(image)
            # concat the attributes
            inference_result = f"age: {age}, gender: {gender}, race: {race}, emotion: {emotion}"
            # Display the caption in the text box
            self.face_textbox.setPlainText(inference_result)
            play_audio(inference_result)

    def describe_environment(self):
        if self.frame is not None:
            # Convert the frame to an RGB image
            image = cv2.cvtColor(self.raw_frame, cv2.COLOR_BGR2RGB)
            caption_text = blip.caption(image,processor = self.blip_processor,model = self.blip_model)
            # Generate the caption using the Blip model
            # is_money = blip.caption(image,processor = self.blip_processor,model = self.blip_model)
            # if is_money == 'yes':
                # money_dict = {}
                # denomination = blip.caption(image,text = 'list all the denominations of money present in the image?',processor = self.blip_processor,model = self.blip_model)
                # denomination = blip.caption(image,text = 'list all the colors of the bills in the image?',processor = self.blip_processor,model = self.blip_model)
                # print(f'various denominations of money present in the image: {denomination}')
                # convert denomination to list by splitting the string with _ character
                # denominations = denomination.split(' ')
                # for den in denominations:
                    # count = blip.caption(image,text = f'how many bank {den} Rupee notes are there in the image?',processor = self.blip_processor,model = self.blip_model)
                    # money_dict[den] = count
                # list out all the denominations and their counts in a string separated by commas
                # money_text = ', '.join([f'{k}rs: x{v}' for k, v in money_dict.items()])
                # caption_text = f'There are {money_text}  bank notes in the image'
            # else:
                # caption_text = 'There is no bank note in the image'

            # Display the caption in the text box
            self.caption_textbox.setPlainText(caption_text)
            play_audio(caption_text)

    def moving_average_smoothing(self,path, window_size=5):
        smoothed_path = []
        for i in range(len(path)):
            start = max(0, i - window_size // 2)
            end = min(len(path), i + window_size // 2 + 1)
            avg_point = np.mean(path[start:end], axis=0).astype(int)
            smoothed_path.append(avg_point)
        return smoothed_path


    def angle_between_vectors(self,v1, v2):
        dot_product = np.dot(v1, v2)
        magnitude_product = np.linalg.norm(v1) * np.linalg.norm(v2)
        angle = np.arccos(dot_product / magnitude_product) * (180 / np.pi)
        return angle

    def direction_from_angle(self,angle, threshold=30):
        if angle < -threshold:
            return "left"
        elif angle > threshold:
            return "right"
        else:
            return "forward"
        
    def start_navigation(self):
        if self.tracking and self.bbox:
            x, y, w, h = [int(v) for v in self.bbox]
            target_point = (x + w // 2, y + h)
            cv2.circle(self.frame, target_point, 5, (0, 0, 255), -1)

            save_path = './Midas/inputs/rgb/'
            cv2.imwrite(save_path + 'frame_gui.png', self.raw_frame)
            midas.run_midas(save_path, self.model, self.transform, self.net_w, self.net_h, self.device)
            depth_map = midas._open_map()
            path_result = pf.path_to_point(target_point, depth_map)

        if path_result is None:
            print('No path found')
        else:
            path = path_result[0]
            depth_map_with_path = path_result[1]

            # Smooth the path using moving average
            smoothed_path = self.moving_average_smoothing(path)

            # Draw the smoothed path on the frame
            for i in range(len(smoothed_path) - 1):
                cv2.line(self.frame, tuple(smoothed_path[i]), tuple(smoothed_path[i + 1]), (0, 0, 255), 2)

                # calculate the distance left on the path
                distance_left = pf.distance_left(path)
                scaled_distance_left = distance_left
                # add the distance left to the navigation button
                self.navigate_button.setText(f'Navigating... ({distance_left:.2f} px)')
                
                # Calculate the direction of the nearest portion of the smoothed path
                origin = np.array([self.frame.shape[1] // 2, self.frame.shape[0]])
                try:
                    nearest_point = np.array(smoothed_path[2])
                except IndexError:
                    print('At the end of the path')
                    nearest_point = np.array(smoothed_path[0])

                forward_direction = np.array([0, -1])  # Assuming the user is facing up in the frame
                path_direction = nearest_point - origin

                angle = self.angle_between_vectors(forward_direction, path_direction)
                cross_product = np.cross(forward_direction, path_direction)
                signed_angle = angle if cross_product >= 0 else -angle

                direction = self.direction_from_angle(signed_angle)

                
                print(f'direction: {direction}')

                # Display the direction on the frame
                if direction:
                    cv2.putText(self.frame, direction.upper(), (origin[0] - 50, origin[1] - 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

                cv2.imshow('Depth Map', depth_map_with_path)
                # cv2.imshow('Navigation', self.frame)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = VideoWidget()
    window.show()
    sys.exit(app.exec_())
