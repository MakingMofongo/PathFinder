import sys
import os
sys.path.append(os.path.join(os.getcwd(), 'Midas'))
import Midas.run as Mid
import path_finding as pf


model = 'dpt_beit_large_512'
Mid.run(None,'./Midas/outputs',f'./Midas/weights/{model}.pt',model_type=model,grayscale=True)



# add ./Midas to sys.path