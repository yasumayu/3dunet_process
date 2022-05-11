import h5py
import glob
import io
import os
import os.path
import sys
import numpy as np
from PIL import Image

input_raw_path = sys.argv[1]

#ラベルデータのパス入力
input_actin_path = sys.argv[2]
input_myosin_path = sys.argv[3]
input_colloid_path = sys.argv[4]
input_mitochon_path = sys.argv[5]

#生データのディレクトリ名の取得(.tiffを除いたもの)
name = os.path.basename(input_raw_path)[10:11]
h5_path = f'./multi_{name}.h5'
#h5_path = "./dualtomo.h5"

"""""
(name).h5
├── raw
├── label ──├── actin
            ├── myosin
            ├── colloid
            ├── mitochon
"""

#HDF5ファイルの作成
with h5py.File(h5_path, "w") as f:

    #生データのtiffファイルを読み込み
    rawdata = np.array(Image.open(input_raw_path))
    raw_datasets  = f.create_dataset(name='raw', data=rawdata, dtype=rawdata.dtype)

    label = f.create_group('label')
    
    #アクチンのアノテーションを行ったtiffファイルの読み込み
    actindata = np.array(Image.open(input_actin_path))
    actin_datasets = f.create_dataset('label/actin', data=actindata, dtype=actindata.dtype)

    #ミオシンのアノテーションを行ったtiffファイルの読み込み
    myosindata = np.array(Image.open(input_myosin_path))
    myosin_datasets = f.create_dataset('label/myosin', data=myosindata, dtype=myosindata.dtype)

    #金コロイドのアノテーションを行ったtiffファイルの読み込み
    colloiddata = np.array(Image.open(input_colloid_path))
    colloid_datasets = f.create_dataset('label/colloid', data=colloiddata, dtype=colloiddata.dtype)

    #ミトコンドリアのアノテーションを行ったtiffファイルの読み込み
    mitochondata = np.array(Image.open(input_mitochon_path))
    mitochon_datasets = f.create_dataset('label/mitochon', data=mitochondata, dtype=mitochondata.dtype)



    

    
