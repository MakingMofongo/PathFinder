import torch
import cv2
import time
import keyboard
import sys
import os





def load_yolo():
    # load yolo v5 model
    model = torch.hub.load("ultralytics/yolov5","yolov5s",pretrained = True)
    return model


def inference(img = None, Loop = True,model = None):
    while True:
        while (img is None):
            img = cv2.imread('./Midas/outputs/rgb/2.png', cv2.IMREAD_COLOR)
        img_gpt = img.copy()

        results = model(img_gpt)
        objects=[]
        # append object names to list
        for i in range(len(results.pandas().xyxy[0]["name"].values[:])):
            objects.append(results.pandas().xyxy[0]["name"].values[i])
        # append bounding box locations to list
        xyxy = results.pandas().xyxy[0]
        xmins = xyxy.xmin.values[:]
        ymins = xyxy.ymin.values[:]
        xmaxs = xyxy.xmax.values[:]
        ymaxs = xyxy.ymax.values[:]
        locations = []
        for i in range(len(xmins)):
            locations.append([xmins[i],ymins[i],xmaxs[i],ymaxs[i]])
            # keep 2 decimal places
            locations[i] = [round(j,2) for j in locations[i]]

        # print (locations)
        # print (objects)

        results.render()

        # cv2.imshow("ObjectDetection",img_gpt)
        # cv2.waitKey(1)

        if(keyboard.is_pressed('esc') == True):
            return 0
        
        if(Loop == False):
            return [objects,locations,img_gpt]
        
        
if __name__ == '__main__':
    # inference(Loop=False)
    img = inference(Loop=False)[2]
    cv2.imshow("ObjectDetection",img)
    cv2.waitKey(0)