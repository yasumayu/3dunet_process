import os
import os.path
import sys
from this import s
import h5py
from matplotlib import pyplot as plt
import numpy as np
import scipy as sp
import tifffile as tif


def h5read(input_h5_path):

    f = h5py.File((input_h5_path), 'r')
    pre_dset = f['predictions']

    channel, z_num, x_num, y_num = map(int, pre_dset.shape)
    print(pre_dset.shape)

    return z_num, x_num, y_num, pre_dset

def h52tiff(channel, z_num, x_num, y_num, dset):

    slice_step = 1
    img = np.zeros((z_num, x_num, y_num))
    for i in range(0, z_num, slice_step):
        img[i][0:x_num][0:y_num] += dset[channel][i][0:x_num][0:y_num]
    
    return img


def main():

    input_h5_dir = sys.argv[1]     
    
 
    z, x, y, dset = h5read(input_h5_dir)

    actin_img = np.zeros(1)
    actin_img = h52tiff(0, z, x, y, dset)
    tif.imsave(f'multi_actin_predictions.tif', actin_img)
                
    myosin_img = np.zeros(1)
    myosin_img = h52tiff(1, z, x, y, dset)
    tif.imsave(f'multi_myosin_predictions.tif', myosin_img)
                
    colloid_img = np.zeros(1)
    colloid_img = h52tiff(2, z, x, y, dset)
    tif.imsave(f'multi_colloid_predictions.tif', colloid_img)

    mitochon_img = np.zeros(1)
    mitochon_img = h52tiff(3, z, x, y, dset)
    tif.imsave(f'multi_mitochon_predictions.tif', mitochon_img)
        



if __name__ == '__main__':
    main()
