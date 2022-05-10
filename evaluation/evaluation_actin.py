import sys

import numpy as np
import tifffile as tif

# data path input. read
input_testdata_path = sys.argv[1]
input_predictiondata_path = sys.argv[2]
test = np.array(tif.imread(input_testdata_path))
pred = np.array(tif.imread(input_predictiondata_path))

z_test, x_test, y_test = map(int, test.shape)
# print(f'true z,x,y : {z_true, x_true, y_true}')
z_pred, x_pred, y_pred = map(int, pred.shape)
# print(f'pred z,x,y : {z_pred, x_pred, y_pred}'

# binalization
thresh = int(sys.argv[3])
print(f'threshold:{thresh}')

pred_thresh = pred > thresh
# print(pred_thresh)

# calcuate confusion matrix
step = 1
tp, fp, tn, fn = 0, 0, 0, 0

for i in range(0, z_test, step):
    for j in range(0, x_test, step):
        for k in range(0, y_test, step):
            if pred_thresh[i][j][k] == 1 and test[i][j][k] == 1:
                tp = tp + 1

            elif pred_thresh[i][j][k] == 0 and test[i][j][k] == 1:
                fn = fn + 1

            elif pred_thresh[i][j][k] == 1 and 2 <= test[i][j][k] <= 4:
                fp = fp + 1

            elif pred_thresh[i][j][k] == 0 and 2 <= test[i][j][k] <= 4:
                tn = tn + 1

print(f'tn:{tn}')
print(f'fp:{fp}')
print(f'fn:{fn}')
print(f'tp:{tp}')

# Precision Precision =  TP / (TP + FP)
precision = tp / (tp + fp)
print(f'Precision:{precision}')

# Recall Recall = TP / (TP + FN)
recall = tp / (tp + fn)
print(f'Recall:{recall}')

# IoU IoU = TP/(TP+FP+FN)
iou = tp / (tp + fp + fn)
print(f'IoU:{iou}')

# f1 score f1 = (2*TP)/(2*TP+FP+FN)
f1 = (2 * tp) / (2 * tp + fp + fn)
print(f'f1:{f1}')
