import torch
import cv2
import time
import keyboard
# load yolo v5 model
model = torch.hub.load("ultralytics/yolov5","yolov5s",pretrained = True)
def inference(img = None, Loop = True):

    while True:
        while (img is None):
            img = cv2.imread('./Midas/outputs/rgb/2.png', cv2.IMREAD_COLOR)
        img_gpt = img.copy()
        results = model(img_gpt)
        # print (results.pandas().xyxy[0])
        objects=[]
        # append object names to list
        for i in range(len(results.pandas().xyxy[0]["name"].values[:])):
            objects.append(results.pandas().xyxy[0]["name"].values[i])
        # append bounding box locations to list
        xmins = results.pandas().xyxy[0]["xmin"].values[:]
        ymins = results.pandas().xyxy[0]["ymin"].values[:]
        xmaxs = results.pandas().xyxy[0]["xmax"].values[:]
        ymaxs = results.pandas().xyxy[0]["ymax"].values[:]
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
    inference(Loop=False)
    # img = inference(Loop=False)
    # cv2.imshow("ObjectDetection",img)
    # cv2.waitKey(0)