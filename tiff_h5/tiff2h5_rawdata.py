import h5py
import glob
import io
import os
import os.path
import sys
import numpy as np
import tifffile as tif

# ラベルデータ,生データのパス入力
input_raw_path = sys.argv[1]
# 生データのディレクトリ名の取得(.tiffを除いたもの)
name = os.path.basename(input_raw_path)[0:-5]
h5_path = f'./{name}.h5'
# h5_path = "./dualtomo.h5"

"""""
(name).h5
├──raw

"""

# HDF5ファイルの作成
with h5py.File(h5_path, "w") as f:
    # 生データのtiffファイルを読み込み
    rawdata = np.array(tif.imread(input_raw_path))
    raw_datasets = f.create_dataset(name='raw', data=rawdata, dtype=rawdata.dtype)


