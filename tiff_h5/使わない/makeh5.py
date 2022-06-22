import h5py
import glob
import io
import os
import os.path
import sys
import numpy as np
from PIL import Image

#ラベルデータ,生データのパス入力
input_label_path = sys.argv[1]
#input_raw_path = sys.argv[2]
#生データのディレクトリ名の取得(.tiffを除いたもの)
#name = os.path.basename(input_raw_path)[0:-5]
#h5_path = f'./{name}.h5'
h5_path = "./dualtomo.h5"

"""""
(name).h5
├── label
│ 
├──raw

"""

#HDF5ファイルの作成
with h5py.File(h5_path, "w") as f:
    
    #ラベルデータのtiffファイルの読み込み
    labeldata = np.array(Image.open(input_label_path))
    label_datasets = f.create_dataset(name='label', data=labeldata, dtype=labeldata.dtype)


    #生データのtiffファイルを読み込み
    #rawdata = np.array(Image.open(input_raw_path))
    #raw_datasets  = f.create_dataset(name='raw', data=rawdata, dtype=rawdata.dtype)
    


   
