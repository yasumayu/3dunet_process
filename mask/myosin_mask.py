import sys
from tkinter import Y
import numpy as np
import tifffile as tif

#ミオシンの閾値処理データから。ミトコンドリア・金コロイドを除く処理
#file読み込み
def file_read(input_path):
    data = np.array(tif.imread(input_path))
    print(data.shape)

    return data

def mask(data1, data2):
    z_test, y_test, x_test = map(int, data1.shape)     
     
    for i in range(0,z_test):
        for j in range(0,y_test):
            for k in range(0,x_test):
                if data1[i][j][k] == 255 and data2[i][j][k] ==255:
                    data1[i][j][k] = 0 
                    print(data1[i][j][k])
                    
    
    return data1


def main():
    #data_path input, read
    input_myosin = sys.argv[1]
    input_colloid = sys.argv[2]
    input_mitochon = sys.argv[3]

    myosin_data = file_read(input_myosin)
    colloid_data = file_read(input_colloid)
    mitochon_data = file_read(input_mitochon)

    #ミオシンのpredictionから金コロイドを外す
    mask_colloid = mask(myosin_data, colloid_data)

    #ミオシンのpredictionからミトコンドリアを外す
    mask_mitochon = mask(mask_colloid, mitochon_data)

    #マスク処理した画像を出力
    tif.imsave('myosin_mask.tif', mask_mitochon)




if __name__ == '__main__':
    main()