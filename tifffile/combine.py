from email.mime import image
import numpy as np
import os
import os.path
import sys
import tifffile as tif
from tifffile.tifffile import imread


scope_dir = sys.argv[1]
#ファイル名(文字列)のリスト
file_list = os.listdir(path=scope_dir)
# print(file_list)
tiff_file_list = []
for f in file_list:
    if '.tif' in f or '.tiff' in f:
        tiff_file_list.append(scope_dir+f)

tiff_file_list.sort()
print(tiff_file_list)

#データの読み込み
split_img = []
i = 0
for tf in tiff_file_list:
    split_img.append(np.array(tif.imread(tf)))
    print(split_img[i].shape)
    i += 1


split_num = len(split_img)
split_step = 3

v_split_img = []
k = 0
for j in range(0, split_num, split_step):
    v_split_img.append(np.concatenate([split_img[j], split_img[j+1], split_img[j+2]], axis=2))
    #print(v_split_img[k])
    k += 1
    


img = np.concatenate(v_split_img, axis=1)
print(img.shape)

#tiffで出力
scope_dir = scope_dir if scope_dir[-1] != '/' else scope_dir[:-1] 
name = scope_dir.split('/')[-1]
tif.imwrite(f'{scope_dir}/{name}.tif',img)









   
