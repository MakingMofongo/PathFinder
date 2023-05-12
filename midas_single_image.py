import sys
import os
sys.path.append(os.path.join(os.getcwd(), 'Midas'))
import Midas.run as Mid
import cv2
import path_finding as pf

def run_midas(input):   
    model = 'dpt_swin2_tiny_256'

    Mid.run(input,'./Midas/outputs/forGUI',f'./Midas/weights/{model}.pt',model_type=model,grayscale=True)

def _open_map():
    # open the map
    map = cv2.imread('./Midas/outputs/forGUI/frame.png', cv2.IMREAD_GRAYSCALE)
    return map 

def path(point):
    map = _open_map()
    pf.main(map)

if __name__ == '__main__':
    run_midas('./Midas/inputs/rgb/')
    map = _open_map()
    cv2.imshow('map', map)
    cv2.waitKey(0)
