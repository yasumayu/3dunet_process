from datetime import datetime
from re import A
import sys
from tkinter import TOP
import numpy as np
import tifffile as tif
import datetime

"""
ミトコンドリアが多く含まれてるファイルを除くファイルで計算
0 1 2
3 4 5
6 7 8
"""


def file_read(input_test_dir, input_pre_dir, index):
    test = np.array(tif.imread(f'{input_test_dir}split_R_seg_{index}.tiff'))
    pred = np.array(tif.imread(f'{input_pre_dir}split_actin_MeanIOU_prediction_{index}.tiff'))

    return test, pred


def count(test, pred, thresh_s, thresh_e):

    z_test, x_test, y_test = map(int, test.shape)
    z_pred, x_pred, y_pred = map(int, pred.shape)
    step = 2

    pred_thresh = np.zeros(pred.shape)
    
    tp = np.zeros(thresh_e + 1)
    fp = np.zeros(thresh_e + 1)
    fn = np.zeros(thresh_e + 1)
    tn = np.zeros(thresh_e + 1)
    print(tp.shape)

    for i in range(40, 50):
        for j in range(300, 350):
            for k in range(300, 350):
                for l in range (thresh_s, thresh_e, step):
              
                    pred_thresh[i][j][k] = pred[i][j][k] > l

                    #print(pred_thresh[i][j][k])
                    #print(test[i][j][k])
                    if pred_thresh[i][j][k] == 1.0 and test[i][j][k] == 1:

                        #print(f'tpOK')
                        tp_count = tp[l]
                        #print(tp_count)
                        tp_count += 1
                        #print(tp_count =1)
                        tp[l] = tp_count

                    elif pred_thresh[i][j][k] == 0.0 and test[i][j][k] == 1:

                        #print(f'fnOK')
                        fn_count = fn[l]
                        fn_count += 1
                        fn[l] = fn_count

                    elif pred_thresh[i][j][k] == 1.0 and 2 <= test[i][j][k] <= 4:
                         
                        #print(f'fpOK') 
                        fp_count = fp[l]
                        fp_count += 1
                        fp[l] = fp_count
                        
                    elif pred_thresh[i][j][k] == 0.0 and 2 <= test[i][j][k] <= 4:
                        
                        #print(f'tnOK')
                        tn_count = tn[l]
                        tn_count += 1
                        tn[l] = tn_count

    return tp, fn, fp, tn


def evaluate(sum_tp, sum_tn, sum_fp, sum_fn, thresh_e):

    precision = np.zeros(thresh_e + 1)
    recall  = np.zeros(thresh_e + 1)
    iou = np.zeros(thresh_e + 1)
    f1 = np.zeros(thresh_e + 1)
    f1_3 = np.zeros(thresh_e + 1)
    f1_5 = np.zeros(thresh_e + 1)

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

    #f1_5 score f1_5 = (5*TP)/(5*TP+FP+FN)
    f1_5 = (5 * sum_tp) / (5 * sum_tp + sum_fp + sum_fn)
    print(f'f1_5:{f1_5}')

    return precision, recall, iou, f1, f1_3, f1_5


def main():
    # data path input. read
    input_test_dir = sys.argv[1]
    input_pre_dir = sys.argv[2]
    thresh_s = int(sys.argv[3])
    thresh_e = int(sys.argv[4])
    index_list = list(map(int, sys.argv[5:]))

    dt_st = datetime.datetime.now()
    print(f'start:{dt_st}')

    tp_sum = np.zeros(thresh_e + 1)
    fp_sum = np.zeros(thresh_e + 1)
    fn_sum = np.zeros(thresh_e + 1)
    tn_sum = np.zeros(thresh_e + 1)

    for index in index_list:

        test, pred = file_read(input_test_dir, input_pre_dir, index)
        tp, fn, fp, tn = count(test, pred, thresh_s, thresh_e)

        tp_sum += tp
        fp_sum += fp
        fn_sum += fn
        tn_sum += tn

        print(f'tp:{tp_sum[200]}')
        print(f'tn:{tn_sum[200]}')
        print(f'fn:{fn_sum[200]}')
        print(f'fp:{fp_sum[200]}')

    precision, recall, iou, f1, f1_3, f1_5 = evaluate(tp_sum, tn_sum, fp_sum, fn_sum, thresh_e)

    for i in range(thresh_s, thresh_e, 2):
        with open('threshold_actin_test.txt', 'a') as f:

            f.write(f'Threshold:{i}\n')
            f.write(f'TN:{tn_sum[i]} ')
            f.write(f'TP:{tp_sum[i]} ')
            f.write(f'FN:{fn_sum[i]} ')
            f.write(f'FP:{fp_sum[i]}\n')
            f.write(f'Precision:{precision[i]} ')
            f.write(f'Reacall:{recall[i]} ')
            f.write(f'F1:{f1[i]} ')
            f.write(f'IoU:{iou[i]} ')
            f.write(f'F1_3:{f1_3[i]} ')
            f.write(f'f1_5:{f1_5[i]}\n')
    
    dt_end = datetime.datetime.now()
    print(f'finish:{dt_end}')



if __name__ == '__main__':
    main()
