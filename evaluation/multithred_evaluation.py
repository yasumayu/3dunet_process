import numpy as np
import sys
import tifffile as tif
from concurrent.futures.thread import ThreadPoolExecutor
from tqdm import tqdm
import time

#actin
#値おかしかった

class ImageData:
    def __init__(self):
        self.test = None
        self.pred = None
        self.x_test = 0
        self.y_test = 0
        self.z_test = 0


    # データのパス入力,読み込み
    def read(self):
        input_testdata_path = sys.argv[1]
        input_predictiondata_path = sys.argv[2]

        self.test = np.array(tif.imread(input_testdata_path))
        self.pred = np.array(tif.imread(input_predictiondata_path))

        self.z_test, self.x_test, self.y_test = map(int, self.test.shape)


class Counter:
    def __init__(self, image_data, si, ei):
        self.image_data = image_data

        self.start_index = si
        self.end_index = image_data.z_test if ei > image_data.z_test else ei

        self.tn = 0
        self.fp = 0
        self.fn = 0
        self.tp = 0

        self.precision = 0
        self.recall = 0
        self.iou = 0
        self.f1 = 0

    def result(self):
        return self.tn, self.fp, self.fn, self.tp


    def count(self):
        thresh = int(sys.argv[3])
        pred_thresh = self.image_data.pred > thresh

        for i in tqdm(range(self.start_index, self.end_index), desc=f'{str(self.start_index).zfill(3)}~ (z)', colour='green'):
            for j in tqdm(range(0, self.image_data.x_test), desc=f'{str(self.start_index).zfill(3)}~ (y)', leave=False):
                for k in range(0, self.image_data.y_test):
                    if pred_thresh[i][j][k] == 1 and self.image_data.test[i][j][k] == 1:
                        self.tp = self.tp + 1

                    elif pred_thresh[i][j][k] == 0 and self.image_data.test[i][j][k] == 1:
                        self.fn = self.fn + 1

                    elif pred_thresh[i][j][k] == 1 and 2 <= self.image_data.test[i][j][k] <= 4:
                        self.fp = self.fp + 1

                    elif pred_thresh[i][j][k] == 0 and 2 <= self.image_data.test[i][j][k] <= 4:
                        self.tn = self.tn + 1
        return self


class Evaluation:
    def __init__(self):
        self.sum_tn = 0
        self.sum_fp = 0
        self.sum_fn = 1
        self.sum_tp = 1

        self.precisio = 0
        self.recall = 0
        self.iou = 0
        self.f1 = 0


    def result(self):
        print(f'Precision:{self.precision}')
        print(f'Recall:{self.recall}')
        print(f'IoU:{self.iou}')
        print(f'f1:{self.f1}')


    def add(self, tn, fp, fn, tp):
        self.sum_tn += tn
        self.sum_fp += fp
        self.sum_fn += fn
        self.sum_tp += tp



    def evaluate(self):
        # Precision Precision =  TP / (TP + FP)
        self.precision = self.sum_tp / (self.sum_tp + self.sum_fp)

        # Recall Recall = TP / (TP + FN)
        self.recall = self.sum_tp / (self.sum_tp + self.sum_fn)
        
        # IoU IoU = TP/(TP+FP+FN)
        self.iou = self.sum_tp / (self.sum_tp + self.sum_fp + self.sum_fn)
        
        # f1 score f1 = (2*TP)/(2*TP+FP+FN)
        self.f1 = (2 * self.sum_tp) / (2 * self.sum_tp + self.sum_fp + self.sum_fn)
        


def main():
    image_data = ImageData()
    image_data.read()

    workers = 8
    # split_range = image_data.z_test // workers + 1
    split_range = image_data.z_test // workers + 1

    futures = []
    with ThreadPoolExecutor() as executor:
        for i in range(workers):
            si, ei = split_range * i, split_range * (i+1)
            counter = Counter(image_data=image_data, si=si, ei=ei)
            future = executor.submit(counter.count)
            futures.append(future)
        

    eva = Evaluation()
    for future in futures:
        result = future.result()
        tn, fp, fn, tp = result.result()
        eva.add(tn, fp, fn, tp)
    
    eva.evaluate()
    eva.result()
            


if __name__ == '__main__':
    main()
