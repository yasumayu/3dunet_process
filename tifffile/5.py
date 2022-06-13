import numpy as np
import os
import os.path
import sys
from sklearn.feature_extraction import img_to_graph
import tifffile as tif
from tifffile.tifffile import imread

"""
マルチチャンネル用の4次元tifffileの作成
channnel x Z x X x Y
"""

input_data_path = sys.argv[1] 
img = tif.imread(input_data_path)
img = np.array(img)
print(img.shape)
z_test, x_test, y_test = map(int, img.shape)



step = 1
for i in range(0, z_test, step):
    for j in range(0, x_test, step):
        for k in range(0, y_test, step):

            if img[i][j][k] == 5:

                img[i][j][k] == 0
        
            img[i][j][k] == 1


name = os.path.basename(input_data_path)
tif.imwrite(f'./1.tiff', img)
