import sys
from tkinter import Y
import numpy as np
import tifffile as tif


def file_read(input_data):
    data = np.array(tif.imread(input_data))
    return data


def subtraction(data1, data2):
    z_test, x_test, y_test = map(int, data1.shape)
    sub_data = data1

    for i in range(0, z_test):
        for j in range(0, y_test):
            for k in range(0, x_test):

                if data1[i][j][k] ==255 and data2[i][j][k] == 255:
                    sub_data[i][j][k] = 0

    
    return sub_data


def main():
    
    input_data1 = sys.argv[1]
    input_data2 = sys.argv[2]


    data1 = file_read(input_data1)
    data2 = file_read(input_data2)


    sub_data =  subtraction(data1, data2)

    tif.imsave(f'sub.tif', sub_data)


if __name__ == '__main__':
    main()
