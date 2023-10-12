import sys
from tkinter import Y
import numpy as np
import tifffile as tif

"""
マニュアルで使えるように変更
引数１：ミオシン
引数２：アクチン
引数３：金コロ
引数４：ミトコンドリア
引数5：background
"""


def file_read(input_data1, input_data2,input_data3, input_data4, input_data5):
    myosin_data = np.array(tif.imread(input_data1))
    #print(data.shape)
    actin_data = np.array(tif.imread(input_data2))
    gold_data = np.array(tif.imread(input_data3))
    mitochon_data = np.array(tif.imread(input_data4))
    back_data = np.array(tif.imread(input_data5))
    return myosin_data, actin_data, gold_data, mitochon_data, back_data


def overay(data1, data2, data3,data4,data5):
    z_test, x_test, y_test = map(int, data1.shape)
    data = data1

    for i in range(0, z_test):
        for j in range(0, y_test):
            for k in range(0, x_test):

                if data4[i][j][k] == 255:
                    data[i][j][k] = 4
                elif data3[i][j][k] == 255:
                    data[i][j][k] = 3
                elif data1[i][j][k] == 255:
                    data[i][j][k] = 2 
                elif data2[i][j][k] == 255:
                    data[i][j][k] = 1
                elif data5[i][j][k] == 255:
                    data[i][j][k] = 5
    
    return data


def main():
    
    input_data1 = sys.argv[1]
    input_data2 = sys.argv[2]
    input_data3 = sys.argv[3]
    input_data4 = sys.argv[4]
    input_data5 = sys.argv[5]

    data1, data2, data3,data4, data5 = file_read(input_data1, input_data2, input_data3, input_data4,input_data5)

    data =  overay(data1, data2, data3, data4, data5)

    tif.imsave(f'seg.tif', data)


if __name__ == '__main__':
    main()
