import numpy as np
import os
import os.path
import sys
import tifffile as tif
from tifffile.tifffile import imread

def file_read(input_data):
    img = tif.imread(input_data)
    img = np.array(img)
    print(img.shape)
    return img

def resize(img,x_min,x_max,y_min,y_max):
    z_num, y_num, x_num = map(int, img.shape)
    resize_img = img[0:z_num, int(y_min):int(y_max), int(x_min):int(x_max)]
    return resize_img

def main():
    #データのパス入力,読み込み
    input_data= sys.argv[1] 
    x_min = sys.argv[2]
    x_max = sys.argv[3]
    y_min = sys.argv[4]
    y_max = sys.argv[5]

    data_array = file_read(input_data)

    resize_img = resize(data_array, x_min, x_max, y_min, y_max)
    tif.imwrite('./resize.tiff', resize_img)


if __name__ == '__main__':
    main()