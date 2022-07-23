from re import X
import numpy as np
import os
import os.path
import sys
from sklearn.feature_extraction import img_to_graph
import tifffile as tif
from tifffile.tifffile import imread
import cv2

input_data_path = sys.argv[1] 
axis = int(sys.argv[2])
img = tif.imread(input_data_path)
img_3d = np.array(img)

img_2d_x = img_3d[:,axis,:]
img_2d_y = img_3d[:,:,axis]

cv2.imwrite(f'xslice_{axis}.png',img_2d_x)
cv2.imwrite(f'yslice_{axis}.png',img_2d_y)


    

