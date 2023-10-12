import sys
from tkinter import Y
import numpy as np
import tifffile as tif

"""
221x1899x1899のファイルで計算
threshold決めた後のtifファイルで、他の要素とかぶっている部分を消す。
0:null
1:actin
2:myosin
3,4:gold,nnitochon
"""

def file_read(input_data_path):
    data = np.array(tif.imread(input_data_path))
    print(data.shape)

    return data

def compare(thresh, anno):
    z_test, x_test, y_test = map(int, thresh.shape)

    for i in range(0, z_test):
        for j in range(0, x_test):
            for k in range(0, y_test):
                if thresh[i][j][k] == 255 and anno[i][j][k] == 1:
                    thresh[i][j][k] = 1
                
                elif thresh[i][j][k] == 255 and anno[i][j][k] == 0:
                    thresh[i][j][k] = 1
                
                elif thresh[i][j][k] == 0 and anno[i][j][k] == 1:
                    thresh[i][j][k] = 1
                    
                elif thresh[i][j][k] == 255 and anno[i][j][k] == 2:
                    thresh[i][j][k] = 2
                
                elif thresh[i][j][k] == 0 and anno[i][j][k] == 2:
                    thresh[i][j][k] = 2
                
                elif thresh[i][j][k] == 255 and anno[i][j][k] == 3:
                    thresh[i][j][k] = 3

                elif thresh[i][j][k] ==0 and anno[i][j][k] == 3:
                    thresh[i][j][k] = 3
                
                elif thresh[i][j][k] == 255 and anno[i][j][k] == 4:
                    thresh[i][j][k] = 4
                
                elif thresh[i][j][k] == 0 and anno[i][j][k] == 4:
                    thresh[i][j][k] = 4

    return thresh

def main():
    # data path input. read
    input_thresh_path = sys.argv[1]
    input_anno_path = sys.argv[2]

    #データの読み込み
    thresh_data = file_read(input_thresh_path)
    anno_data = file_read(input_anno_path)

    compare_actin = compare(thresh_data, anno_data)

    tif.imwrite('actin_seg.tif', compare_actin)


if __name__ == '__main__':
    main()



    