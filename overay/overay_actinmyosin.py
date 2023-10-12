import sys
from tkinter import Y
import numpy as np
import tifffile as tif

"""
マニュアルで使えるように変更
引数１：ミオシン
引数２：アクチン
"""


def file_read(input_data1, input_data2):
    myosin_data = np.array(tif.imread(input_data1))
    #print(data.shape)
    actin_data = np.array(tif.imread(input_data2))
    return myosin_data, actin_data


def overay(data1, data2):
    z_test, x_test, y_test = map(int, data1.shape)
    data = data1

    for i in range(0, z_test):
        for j in range(0, y_test):
            for k in range(0, x_test):

                if data1[i][j][k] == 255:
                    data[i][j][k] = 2
                elif data2[i][j][k] == 255:
                    data[i][j][k] = 1
    
    return data


def main():
    
    input_data1 = sys.argv[1]
    input_data2 = sys.argv[2]

    data1, data2 = file_read(input_data1, input_data2)

    data =  overay(data1, data2)

    tif.imsave(f'seg.tif', data)


if __name__ == '__main__':
    main()
