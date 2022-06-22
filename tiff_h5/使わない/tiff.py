import cv2
import os
import os.path
import sys
import numpy as np
import mrcfile

input_raw_path = sys.argv[1]

mmap = mrcfile.asarray(out="memmap") # out="memmap"でメモリマップを返してくれる
#image = cv2.imread(input_raw_path)
print(mmap.shape)

