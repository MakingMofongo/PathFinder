import sys
import os
import numpy as np
sys.path.append(os.path.join(os.getcwd(), 'Midas'))
import Midas.run as Mid
import path_finding as pf
import multiprocessing as mp
import cv2
import time

def on_read_midas_map():
    initial_bgr_image = cv2.imread('./Midas/outputs/rgb/frame.png')
    predictor = sam.setup_sam_model() #only use new depth maps
    last_point = None    
    while True:
        depth_map = None
        
        # read the depth map
        while (depth_map is None or bgr_image is None):
            depth_map = cv2.imread('./Midas/outputs/depth/frame.png', cv2.IMREAD_GRAYSCALE)
            bgr_image = cv2.imread('./Midas/outputs/rgb/frame.png')

        predictor.set_image(bgr_image)
        # read point.txt from depth folder
        point = open('./Midas/outputs/depth/point.txt', 'r').read()
        point = point.split(',')
        if point[0] == '' or point[1] == '':
            continue
        point = [[int(point[0]), int(point[1])]]
        point = np.array(point)

        # check if furthest point changed
        if last_point is not None:
            if not np.array_equal(last_point, point):
                print(f'furthest point changed: {point} \n')


        
        segment_output = sam.segment(bgr_image, predictor, point)
        mask = segment_output[0]
        score = segment_output[1]

        # overlay the mask on the image
        BGRimage = bgr_image.copy()
        BGRimage = sam.overlay_mask(BGRimage, mask, 0.5, random_color=False)
        # BGRimage = cv2.cvtColor(BGRimage, cv2.COLOR_RGB2BGR)

        DEPTHimage = depth_map.copy()
        DEPTHimage = cv2.cvtColor(DEPTHimage, cv2.COLOR_GRAY2BGR)
        # draw a dot at the furthest point
        cv2.circle(DEPTHimage, point[0], 10, (0,255,0), -1)
                
        # Display the image
        cv2.imshow('image', BGRimage)
        cv2.imshow('depth', DEPTHimage)
        cv2.waitKey(1)



        last_point = point
        time.sleep(.1)

        # delete the depth map

# do multiprocessing for path_finding.main and Mid.run
if __name__ == '__main__':

    multi_processor = mp.Process(target=pf.loop_main, args=())
    multi_processor.start()
    # model = 'dpt_beit_large_512'
    model = 'dpt_swin2_tiny_256'
    # multi_processor = mp.Process(target=Mid.run, args=(None,'./Midas/outputs',f'./Midas/weights/{model}.pt',model), kwargs=dict(grayscale=True))
    # multi_processor.start()

    # on_read_midas_map()
    Mid.run(None,'./Midas/outputs',f'./Midas/weights/{model}.pt',model_type=model,grayscale=True)





# add ./Midas to sys.path