import sys
import numpy as np
import tifffile as tif

input_data_path = sys.argv[1]
img = tif.imread(input_data_path)
print(img.shape)

