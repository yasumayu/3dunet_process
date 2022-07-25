import sys
import numpy as np
import tifffile as tif

#221x1899x1899のファイルで計算


def file_read(input_test_path, input_pre_path):
    test = np.array(tif.imread(input_test_path))
    pred = np.array(tif.imread(input_pre_path))

    return test, pred


def count(test, pred, thresh_s, thresh_e, step):
    z_test, x_test, y_test = map(int, test.shape)

    tp = np.zeros(thresh_e)
    tn = np.zeros(thresh_e)
    fp = np.zeros(thresh_e)
    fn = np.zeros(thresh_e)

    pred_thresh = np.zeros((z_test, x_test, y_test))

    print(pred_thresh.shape)

    for i in range(0, z_test):
        for j in range(0, x_test):
            for k in range(0, y_test):
                count_tp = 0
                count_fp = 0
                count_fn = 0
                count_tn = 0
                for th in range(thresh_s, thresh_e, step):
                    pred_thresh = pred > th

                    if pred_thresh[i][j][k] == 1 and test[i][j][k] == 1:
                        count_tp = count_tp + 1
                    elif pred_thresh[i][j][k] == 0 and test[i][j][k] == 1:
                        count_fn = count_fn + 1    
                    elif pred_thresh[i][j][k] == 1 and 2 <= test[i][j][k] <= 4:
                        count_fp = count_fp + 1
                    elif pred_thresh[i][j][k] == 0 and 2 <= test[i][j][k] <= 4:
                        count_tn = count_tn + 1
                
                tp[th] = count_tp
                tn[th] = count_fn
                fp[th] = count_fp
                fn[th] = count_fn
    
    return tp, tn, fp, fn



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

    test, pred = file_read(input_test_path, input_pre_path)

    sum_tp = np.zeros(thresh_e)
    sum_tn = np.zeros(thresh_e)
    sum_fp = np.zeros(thresh_e)
    sum_fn = np.zeros(thresh_e)
    thresh_step = 2

    sum_tp, sum_tn, sum_fp, sum_fn = count(test, pred, thresh_s, thresh_e, thresh_step)

    for i in range(thresh_s, thresh_e, thresh_step):
        precision, recall, iou, f1, f1_3,f1_5  = evaluate(sum_tp[i], sum_tn[i], sum_fp[i], sum_fn[i]) 
        """
        with open('threshold_actin_multi.txt', 'a') as f:
            f.write(f'Threshold:{i}\n')
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
        """



if __name__ == '__main__':
    main()
