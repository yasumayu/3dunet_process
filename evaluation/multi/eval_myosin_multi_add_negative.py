import sys
from tkinter import Y
import numpy as np
import tifffile as tif

"""
221x1899x1899のファイルで計算
ミオシンフィラメンとの周辺をNegativeに判定する
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
    z_test, x_test, y_test = map(int, test.shape)
    test_add_negative = test

    for i in range(0, z_test-2):
        for j in range(0, y_test-2):
            for k in range(0, x_test-2):
                if test[i][j][k] == 0 and test[i][j][k+1] == 0 and test[i][j][k+2] == 2:
                    test_add_negative[i][j][k] = 5
                    test_add_negative[i][j][k+1] = 5
                
                elif test[i][j][k] == 0 and test[i][j][k+1] == 2 and test[i][j][k+2] == 0:
                    test_add_negative[i][j][k] = 5
                    test_add_negative[i][j][k+2] = 5

                elif test[i][j][k] == 0 and test[i][j][k+1] == 2 and test[i][j][k+2] == 2:
                    test_add_negative[i][j][k] = 5
                
                elif test[i][j][k] == 2 and test[i][j][k+1] == 0 and test[i][j][k+2] == 0:
                    test_add_negative[i][j][k+1] = 5
                    test_add_negative[i][j][k+2] = 5
                
                elif test[i][j][k] == 2 and test[i][j][k+1] == 0 and test[i][j][k+2] == 2:
                    test_add_negative[i][j][k+1] = 5
                    print(test_add_negative[i][j][k+1])

                elif test[i][j][k] == 2 and test[i][j][k+1] == 2 and test[i][j][k+2] == 0:
                    test_add_negative[i][j][k+2] = 5
                    print(test_add_negative[i][j][k+2])


    return test_add_negative


def count(test, pred, thresh):
    z_test, x_test, y_test = map(int, test.shape)
    #z_pred, x_pred, y_pred = map(int, pred.shape)
    tp, fp, tn, fn = 0, 0, 0, 0
    pred_thresh = pred > thresh

    print(np.count_nonzero(pred_thresh == 1))

    for i in range(0, z_test):
        for j in range(0, y_test):
            for k in range(0, x_test):
                if pred_thresh[i][j][k] == 1 and test[i][j][k] == 2:
                    tp = tp + 1
                    #print('tp')

                elif pred_thresh[i][j][k] == 0 and test[i][j][k] == 2:
                    fn = fn + 1
                    #print('fn')

                elif pred_thresh[i][j][k] == 1 and  3 <= test[i][j][k] <= 5:
                    fp = fp + 1
                    #print('fp')
                
                elif pred_thresh[i][j][k] == 1 and test[i][j][k] == 1:
                    fp = fp + 1
                    #print('fp')

                elif pred_thresh[i][j][k] == 0 and 3 <= test[i][j][k] <= 5:
                    tn = tn + 1
                    #print('tn')
                
                elif pred_thresh[i][j][k] == 0 and test[i][j][k] == 1:
                    tn = tn + 1
                    #print('tn')
                
    return tp, fn, fp, tn

def evaluate(sum_tp, sum_tn, sum_fp, sum_fn):

    # Precision Precision =  TP / (TP + FP)
    precision = sum_tp / (sum_tp + sum_fp)
    print(f'Precision:{precision}')

    # Recall Recall = TP / (TP + FN)
    recall = sum_tp / (sum_tp + sum_fn)
    print(f'Recall:{recall}')

    # IoU IoU = TP/(TP+FP+FN)
    iou = sum_tp / (sum_tp + sum_fp + sum_fn)
    print(f'IoU:{iou}')

    # f1 score f1 = (2*TP)/(2*TP+FP+FN)
    f1 = (2 * sum_tp) / (2 * sum_tp + sum_fp + sum_fn)
    print(f'f1:{f1}')

    # f1_3 score f1_3 = (3*TP)/(3*TP+FP+FN)
    f1_3 = (3 * sum_tp) / (3 * sum_tp + sum_fp + sum_fn)
    print(f'f1_3:{f1_3}')

    f1_5 = (5 * sum_tp) / (5 * sum_tp + sum_fp + sum_fn)
    print(f'f1_5:{f1_5}')

    return precision, recall, iou, f1, f1_3, f1_5


def main():
    # data path input. read
    input_test_path = sys.argv[1]
    input_pre_path = sys.argv[2]
    thresh_s = int(sys.argv[3])
    thresh_e = int(sys.argv[4])

    #テストデータの読み込み
    test_data = file_read(input_test_path)

    #テストデータにNegativeを追加
    test = negative(test_data)

    threshold_step = 2
    for thresh in range(thresh_s, thresh_e, threshold_step):

        print(thresh)

        sum_tp, sum_tn, sum_fp, sum_fn = 0, 0, 0, 0

        pred = file_read(input_pre_path)

        sum_tp, sum_fn, sum_fp, sum_tn = count(test, pred, thresh)

        precision, recall, iou, f1, f1_3,f1_5  = evaluate(sum_tp, sum_tn, sum_fp, sum_fn)

        with open('threshold_myosin_multi_add_negative_230809.txt', 'a') as f:
            f.write(f'Threshold:{thresh} ')
            f.write(f'TN:{sum_tn} ')
            f.write(f'TP:{sum_tp} ')
            f.write(f'FN:{sum_fn} ')
            f.write(f'FP:{sum_fp} ')
            f.write(f'Precision:{precision} ')
            f.write(f'Reacall:{recall} ')
            f.write(f'IoU:{iou} ')
            f.write(f'F1:{f1}\n')
           


if __name__ == '__main__':
    main()
