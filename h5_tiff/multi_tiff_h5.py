#使わん
from math import comb
import os
import os.path
import numpy as np
import sys
import h5py
from matplotlib import pyplot as plt
import tifffile as tif

def h5read(input_h5_dir, index):
    f = h5py.File((f'{input_h5_dir}split_R_{index}_predictions.h5'), 'r')

    #予測した画像のデータセットの取得
    dset = f['predictions']
    print(dset.shape)

    channel_num, z_num, x_num, y_num = map(int, dset.shape)

    return z_num, x_num, y_num, dset

#h５ファイルからtiffにファイル変換
def h5totiff(z_num, x_num, y_num , dset): 

    slice_step = 1
    # xを動かしてyz面のスライスの画像を保存
    actin_slice = np.zeros((z_num, x_num, y_num))
    myosin_slice = np.zeros((z_num, x_num, y_num))
    colloid_slice = np.zeros((z_num, x_num, y_num))
    mitochon_slice = np.zeros((z_num, x_num, y_num))

    for i in range(0, z_num, slice_step):
        actin_slice[i][0:x_num][0:y_num] += dset[0][i][0:x_num][0:y_num]
        myosin_slice[i][0:x_num][0:y_num]  += dset[1][i][0:x_num][0:y_num]
        colloid_slice[i][0:x_num][0:y_num]  += dset[2][i][0:x_num][0:y_num]
        mitochon_slice[i][0:x_num][0:y_num]  += dset[3][i][0:x_num][0:y_num]
    
    print(f'actinslice:{actin_slice.shape}')
    return actin_slice, myosin_slice, colloid_slice, mitochon_slice

def combine(img_dict):

    split_step = 3
    split_num = len(img_dict)
    split_img = []

    #k = 0
    for j in range(0, split_num, split_step):
        split_img.append(np.concatenate([img_dict[j], img_dict[j+1], img_dict[j+2]], axis=2))
        #print(v_split_img[k])
        #k += 1
        
    img = np.concatenate(split_img, axis=1)

    print(img.shape)

    return img


def main():

    #h5ファイルの入ったディレクトリの取得
    input_h5_dir = sys.argv[1]

    #辞書
    dict_actin = {}
    dict_myosin = {}
    dict_colloid = {}
    dict_mitochon = {}

    step = 1
    for index in range(0,9,step):

        z, x, y, dset = h5read(input_h5_dir,index)
        #split0~8を辞書型で格納
        dict_actin[index], dict_myosin[index], dict_colloid[index], dict_mitochon[index] = h5totiff(z, x, y, dset)
    
    actin_img=[]
    myosin_img=[]
    colloid_img = []
    mitochon_img = []

    actin_img = combine(dict_actin)
    myosin_img = combine(dict_myosin)
    colloid_img = combine(dict_colloid)
    mitochon_img = combine(dict_mitochon) 
    print(actin_img.shape, myosin_img.shape, colloid_img.shape, mitochon_img.shape)

    tif.imsave(f'multi_actin_prediction.tiff', actin_img)
    tif.imsave(f'multi_myosin_prediction.tiff', myosin_img)
    tif.imsave(f'multi_colloid_prediction.tiff', colloid_img)
    tif.imsave(f'multi_itochon_prediction.tiff', mitochon_img)




if __name__ == '__main__':
    main()