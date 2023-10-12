import numpy as np
import os
import os.path
import sys
import tifffile as tif
from tifffile.tifffile import imread

input_data_path = sys.argv[1] 
img = tif.imread(input_data_path)
print(img.shape)

img_3d = np.zeros((221,1100,1100))
z_test, x_test, y_test = map(int, img_3d.shape)

step = 1
for j in range(0,z_test, step):
       for k in range(0,y_test, step):
              for l in range(0,x_test, step):
                img_3d[j][k][l] = img[j][k][l]

tif.imwrite('3d.tif', img_3d)


