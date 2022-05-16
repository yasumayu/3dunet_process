import h5py
import glob
import io
import os
import os.path
import sys
import numpy as np
import tifffile as tif


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
├── label_actin
├── label_myosin
├── label_colloid
├── label_mitochon
"""

#HDF5ファイルの作成
with h5py.File(h5_path, "w") as f:

    #生データのtiffファイルを読み込み
    rawdata = np.array(tif.imread(input_raw_path))
    raw_datasets  = f.create_dataset(name='raw', data=rawdata, dtype=rawdata.dtype)

    """
    label = f.create_group('label_actin')
    label = f.create_group('label_myosin')
    label = f.cleate_group('label_colloid')
    label = f.create_group('label_mitochon')
    """
    
    #アクチンのアノテーションを行ったtiffファイルの読み込み
    actindata = np.array(tif.imread(input_actin_path))
    actin_datasets = f.create_dataset(name='label_actin', data=actindata, dtype=actindata.dtype)

    #ミオシンのアノテーションを行ったtiffファイルの読み込み
    myosindata = np.array(tif.imread(input_myosin_path))
    myosin_datasets = f.create_dataset(name='label_myosin', data=myosindata, dtype=myosindata.dtype)

    #金コロイドのアノテーションを行ったtiffファイルの読み込み
    colloiddata = np.array(tif.imread(input_colloid_path))
    colloid_datasets = f.create_dataset(name='label_colloid', data=colloiddata, dtype=colloiddata.dtype)

    #ミトコンドリアのアノテーションを行ったtiffファイルの読み込み
    mitochondata = np.array(tif.imread(input_mitochon_path))
    mitochon_datasets = f.create_dataset(name='label_mitochon', data=mitochondata, dtype=mitochondata.dtype)



    

    
