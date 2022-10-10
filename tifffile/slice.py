
import numpy as np
import os
import os.path
import sys
from sklearn.feature_extraction import img_to_graph
import tifffile as tif
from tifffile.tifffile import imread
import cv2


input_data_path = sys.argv[1] 
z_axis = int(sys.argv[2])
y_axis = int(sys.argv[3])
img = tif.imread(input_data_path)
img_3d = np.array(img)

#FIJIが1はじまりで数えられるため、FIJIが１なら切り出す場所は0になる！
xy_slice = img_3d[z_axis-1,:,:]
xz_slice = img_3d[:,y_axis,:]

cv2.imwrite(f'xy_slice_{z_axis}.png',xy_slice)
cv2.imwrite(f'xz_slice_{y_axis}.png',xz_slice)


    