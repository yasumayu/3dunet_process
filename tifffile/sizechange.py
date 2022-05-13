import numpy as np
import os
import os.path
import sys
import tifffile as tif
from tifffile.tifffile import imread

#データのパス入力,読み込み
input_data_path= sys.argv[1] 
img = tif.imread(input_data_path)
img = np.array(img)
print(img.shape)

#x,yをそれぞれ3分割する
v_split = 3
h_split = 3

w_size = img.shape[0]
v_size = img.shape[1] // v_split * v_split
h_size = img.shape[2] // h_split * h_split
img = img[:w_size, :v_size, :h_size]
print(img.shape)

name = os.path.basename(input_data_path).split('.')[0]
tif.imwrite(f'./{name}_seg.tiff', img)




