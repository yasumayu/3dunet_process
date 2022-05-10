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

v_split_size = img.shape[1] // v_split
h_split_size = img.shape[2] // h_split

#分割する
out_img = []
for i in range(v_split):
    for j in range(h_split):
        out_img.append(img[:w_size, i*v_split_size:(i+1)*v_split_size, j*h_split_size:(j+1)*h_split_size])

#tiffで出力
name = os.path.basename(input_data_path).split('.')[0]
out_img_num = len(out_img)
print(out_img_num)
for i in range(out_img_num):
    tif.imwrite(f'./split_{name}_{i}.tiff', out_img[i])







