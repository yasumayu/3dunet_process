import sys
from tkinter import Y
import numpy as np
import tifffile as tif

"""
221x1899x1899のファイルで計算
フィラメンとの周辺をNegativeに判定し、その数をカウント
testについて
0:null
1:actin
2:myosin
3,4:gold,nnitochon
5:negative

"""


def file_read(input_data_path):
    data = np.array(tif.imread(input_data_path))
    print(data.shape)

    return data


def negative(test):
    z_test, y_test, x_test = map(int, test.shape)
    test_add_negative = test
    sum_negative = 0

    for i in range(0, z_test-2, 1):
        print('slice')
        for j in range(0, y_test-2, 1):
            for k in range(0, x_test-2, 3):
                if test[i][j][k] == 0 and test[i][j][k+1] == 0 and test[i][j][k+2] == 1:
                    test_add_negative[i][j][k] = 5
                    test_add_negative[i][j][k+1] = 5

                    sum_negative = sum_negative + 2

                elif test[i][j][k] == 0 and test[i][j][k+1] == 1 and test[i][j][k+2] == 0:
                    test_add_negative[i][j][k] = 5
                    test_add_negative[i][j][k+2] = 5

                    sum_negative = sum_negative + 2

                elif test[i][j][k] == 0 and test[i][j][k+1] == 1 and test[i][j][k+2] == 1:
                    test_add_negative[i][j][k] = 5

                    sum_negative = sum_negative + 1

                elif test[i][j][k] == 1 and test[i][j][k+1] == 0 and test[i][j][k+2] == 0:
                    test_add_negative[i][j][k+1] = 5
                    test_add_negative[i][j][k+2] = 5

                    sum_negative = sum_negative + 2

                elif test[i][j][k] == 1 and test[i][j][k+1] == 0 and test[i][j][k+2] == 1:
                    test_add_negative[i][j][k+1] = 5
                    
                    sum_negative = sum_negative + 1

                elif test[i][j][k] == 1 and test[i][j][k+1] == 1 and test[i][j][k+2] == 0:
                    test_add_negative[i][j][k+2] = 5
                    
                    sum_negative = sum_negative + 1


    return sum_negative


def main():
    # data path input. read
    input_test_path = sys.argv[1]

    #テストデータの読み込み
    test_data = file_read(input_test_path)

    #テストデータにNegativeを追加
    sum_negative = negative(test_data)

    print(sum_negative)


if __name__ == '__main__':
    main()
