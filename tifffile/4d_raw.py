import numpy as np
import os
import os.path
import sys
import tifffile as tif
from tifffile.tifffile import imread

input_data_path = sys.argv[1] 
img = tif.imread(input_data_path)

img_4d = np.zeros((4,221,633,633))
channel,z_test, x_test, y_test = map(int, img_4d.shape)

step = 1
for i in range(0, channel, step):
    for j in range(0,z_test, step):
        for k in range(0,x_test, step):
            for l in range(0,y_test, step):
                img_4d[i][j][k][l] = img[j][k][l]

img_4d = img_4d.astype(int)
print(img_4d.shape)
name = os.path.basename(input_data_path)
tif.imwrite(f'./{name}', img_4d)