import sys
import numpy as np
import tifffile as tif

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


def count(test, pred, thresh):
    z_test, x_test, y_test = map(int, test.shape)
    z_pred, x_pred, y_pred = map(int, pred.shape)

    tp, fp, tn, fn = 0, 0, 0, 0
    pred_thresh = pred > thresh

    for i in range(0, z_test):
        for j in range(0, x_test):
            for k in range(0, y_test):
                if pred_thresh[i][j][k] == 1 and test[i][j][k] == 1:
                    tp = tp + 1

                elif pred_thresh[i][j][k] == 0 and test[i][j][k] == 1:
                    fn = fn + 1

                elif pred_thresh[i][j][k] == 1 and 2 <= test[i][j][k] <= 4:
                    fp = fp + 1

                elif pred_thresh[i][j][k] == 0 and 2 <= test[i][j][k] <= 4:
                    tn = tn + 1

    return tp, fn, fp, tn


def evaluate(sum_tp, sum_tn, sum_fp, sum_fn):

    # f1 score f1 = (2*TP)/(2*TP+FP+FN)
    f1 = (2 * sum_tp) / (2 * sum_tp + sum_fp + sum_fn)
    print(f'f1:{f1}')
    
    f1_3 = (3 * sum_tp) / (3 * sum_tp + sum_fp + sum_fn)
    print(f'f1_3:{f1_3}')

    f1_5 = (5 * sum_tp) / (5 * sum_tp + sum_fp + sum_fn)
    print(f'f1_5:{f1_5}')

    f1_7 = (7 * sum_tp) / (7 * sum_tp + sum_fp + sum_fn)
    print(f'f1_7:{f1_7}')

    f1_9 = (9 * sum_tp) / (9 * sum_tp + sum_fp + sum_fn)
    print(f'f1_9:{f1_9}')

    return  f1, f1_3, f1_5, f1_7, f1_9


def main():
    # data path input. read
    input_test_dir = sys.argv[1]
    input_pre_dir = sys.argv[2]
    thresh_s = int(sys.argv[3])
    thresh_e = int(sys.argv[4])
    index_list = list(map(int, sys.argv[5:]))

    threshold_step = 2
    for thresh in range(thresh_s, thresh_e, threshold_step):

        print(thresh)

        sum_tp, sum_tn, sum_fp, sum_fn = 0, 0, 0, 0

        for index in index_list:

            test, pred = file_read(input_test_dir, input_pre_dir, index)
            tp, fn, fp, tn = count(test, pred, thresh)

            sum_tp += tp
            sum_tn += tn
            sum_fp += fp
            sum_fn += fn
            sum = sum_tp + sum_tn + sum_fp + sum_fn
            print(sum)

        f1, f1_3, f1_5, f1_7, f1_9 = evaluate(sum_tp, sum_tn, sum_fp, sum_fn)
        with open('f1_actin.txt', 'a') as f:
            f.write(f'Threshold:{thresh}\n')
            f.write(f'F1:{f1}\n')
            f.write(f'F1_3:{f1_3}\n')
            f.write(f'F1_5:{f1_5}\n')
            f.write(f'F1_7:{f1_7}\n')
            f.write(f'F1_9:{f1_9}\n')



if __name__ == '__main__':
    main()
