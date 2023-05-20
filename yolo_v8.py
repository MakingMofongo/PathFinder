from ultralytics import YOLO
import torch
import cv2
import time
import keyboard
import sys
import os





def load_yolo():
    # load yolo v5 model
    # model = torch.hub.load("ultralytics/yolov5","yolov5s",pretrained = True)

    model = YOLO("yolov8n.pt")
    return model


def inference(img = None, Loop = True):
    model = load_yolo()
    while True:
        while (img is None):
            img = cv2.imread('./Midas/outputs/rgb/2.png', cv2.IMREAD_COLOR)
        plotted = img.copy()
        # convert img_gpt to ndarray
    
        yolo_output = model(source=plotted)

        classes = yolo_output[0].names # dict of class names with index
        detected_boxes = yolo_output[0].boxes
        object_indexes = detected_boxes.cls
        # convert object_indexes tensor to list
        object_indexes = object_indexes.tolist()
        print (f"object_indexes: {object_indexes}")

        object_names=[]
        bounding_boxes = []

        # append object names to list and bounding boxes to list
        for i in range(len(object_indexes)):
            object_names.append(classes[object_indexes[i]])

            box = detected_boxes[i]
            box = box.xyxy[0].tolist()
            bounding_boxes.append(box)

        plotted = yolo_output[0].plot()

        if(keyboard.is_pressed('esc') == True):
            return 0
        
        if(Loop == False):
            return [object_names,bounding_boxes,plotted]
        
        
if __name__ == '__main__':
    inference(Loop=False)
    # img = inference(Loop=False)
    # cv2.imshow("ObjectDetection",img)
    # cv2.waitKey(0)