import numpy as np
import os
import os.path
import sys
import tifffile as tif
from tifffile.tifffile import imread

"""
0,255の画素値を持つ画像を、0,1に変換
"""
#データのパス入力,読み込み
input_data_path= sys.argv[1] 
name = os.path.basename(input_data_path)[0:-5]

img = tif.imread(input_data_path)
img = np.array(img)
print(img.shape)

img[img < 255] = 0
img[img >= 255] = 1

tif.imwrite(f'{name}.tiff',img)
