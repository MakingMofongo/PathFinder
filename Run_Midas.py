import sys
import os
sys.path.append(os.path.join(os.getcwd(), 'Midas'))
import Midas.run as Mid
import path_finding as pf
import multiprocessing as mp


# do multiprocessing for path_finding.main and Mid.run
if __name__ == '__main__':
    multi_processor = mp.Process(target=pf.loop_main, args=())
    multi_processor.start()
    
    # model = 'dpt_beit_large_512'
    model = 'dpt_swin2_tiny_256'
    Mid.run(None,'./Midas/outputs',f'./Midas/weights/{model}.pt',model_type=model,grayscale=True)




# add ./Midas to sys.path