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


def resize(img,factor=2):
    # resize image to 1/factor of original size
    height,width,_ = img.shape
    img = cv2.resize(img,(int(width/factor),int(height/factor)))
    return img


def inference(img = None,model = None):
    while (img is None):
        img = cv2.imread('./Midas/outputs/rgb/2.png', cv2.IMREAD_COLOR)
    img_gpt = img.copy()
    # img_gpt = resize(img_gpt) 
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
    return [objects,locations,img_gpt]
        
        
if __name__ == '__main__':
    # use yolo on live webcam feed
    model = load_yolo()
    cam = cv2.VideoCapture(0)
    while True:
        start = time.time()
        ret,frame = cam.read()
        i = frame.copy()
        o,l,i = inference(img = frame,model = model)
        # add fps to image
        cv2.putText(i,"FPS: "+str(int(1/(time.time()-start))),(10,30),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2)
        cv2.imshow("ObjectDetection",i)
        cv2.waitKey(1)

        if(keyboard.is_pressed('esc') == True):
            break
    cv2.destroyAllWindows()
    cam.release()


    # # inference(Loop=False)
    # img = inference(Loop=False)[2]
    # cv2.imshow("ObjectDetection",img)
    # cv2.waitKey(0)