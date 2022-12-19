import sys
from tkinter import Y
import numpy as np
import tifffile as tif

"""
221x1899x1899のファイルで計算
フィラメンとの周辺をNegativeに判定する
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

    for i in range(0, z_test):
        for j in range(0, x_test):
            for k in range(0, y_test):
                if test[i][j][k] == 0 and test[i][j+1][k] == 0 and test[i][j+2][k] == 1:
                    test_add_negative[i][j][k] == 5
                    test_add_negative[i][j+1][k] == 5
                
                elif test[i][j][k] == 0 and test[i][j+1][k] == 1 and test[i][j+2][k] == 0:
                    test_add_negative[i][j][k] == 5
                    test_add_negative[i][j+2][k] == 5

                elif test[i][j][k] == 0 and test[i][j+1][k] == 1 and test[i][j+2][k] == 1:
                    test_add_negative[i][j][k] == 5
                
                elif test[i][j][k] == 1 and test[i][j+1][k] == 0 and test[i][j+2][k] == 0:
                    test_add_negative[i][j+1][k] == 5
                    test_add_negative[i][j+2][k] == 5
                
                elif test[i][j][k] == 1 and test[i][j+1][k] == 0 and test[i][j+2][k] == 1:
                    test_add_negative[i][j+1][k] == 5

                elif test[i][j][k] == 1 and test[i][j+1][k] == 1 and test[i][j+2][k] == 0:
                    test_add_negative[i][j+2][k] == 5


    return test_add_negative


def count(test, pred, thresh):
    z_test, x_test, y_test = map(int, test.shape)
    #z_pred, x_pred, y_pred = map(int, pred.shape)
    tp, fp, tn, fn = 0, 0, 0, 0
    pred_thresh = pred > thresh

    print(np.count_nonzero(pred_thresh == 1))

    for i in range(0, z_test):
        for j in range(0, x_test):
            for k in range(0, y_test):
                if pred_thresh[i][j][k] == 1 and test[i][j][k] == 1:
                    tp = tp + 1

                elif pred_thresh[i][j][k] == 0 and test[i][j][k] == 1:
                    fn = fn + 1

                elif pred_thresh[i][j][k] == 1 and 2 <= test[i][j][k] <= 5:
                    fp = fp + 1

                elif pred_thresh[i][j][k] == 0 and 2 <= test[i][j][k] <= 5:
                    tn = tn + 1

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
    test = file_read(input_test_path)

    #テストデータにNegativeを追加




    threshold_step = 2
    for thresh in range(thresh_s, thresh_e, threshold_step):

        print(thresh)

        sum_tp, sum_tn, sum_fp, sum_fn = 0, 0, 0, 0

        pred = file_read(input_pre_path)

        sum_tp, sum_fn, sum_fp, sum_tn = count(test, pred, thresh)

        precision, recall, iou, f1, f1_3,f1_5  = evaluate(sum_tp, sum_tn, sum_fp, sum_fn)

        with open('threshold_actin_multi_add_negative.txt', 'a') as f:
            f.write(f'Threshold:{thresh}\n')
            f.write(f'TN:{sum_tn} ')
            f.write(f'TP:{sum_tp} ')
            f.write(f'FN:{sum_fn} ')
            f.write(f'FP:{sum_fp}\n')
            f.write(f'Precision:{precision} ')
            f.write(f'Reacall:{recall} ')
            f.write(f'IoU:{iou} ')
            f.write(f'F1:{f1} ')
            f.write(f'F1_3:{f1_3} ')
            f.write(f'F1_5:{f1_5}\n')
            


if __name__ == '__main__':
    main()
