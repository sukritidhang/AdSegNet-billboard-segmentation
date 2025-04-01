# Import the required libaries
import os
import glob
import numpy as np
import skimage.io
from skimage import color
import cv2
import natsort
import csv


# Evaluator class.
class Evaluate(object):
    # Class constructor, assigns prediction and ground truth.
    def __init__(self, pred_image, gt_image, image_name):
        self.pred_image = pred_image
        self.gt_image = gt_image
        self.image_name = image_name
    
    def roc(self):

        self.pred_image
        self.gt_image
               
        TP = TN = FP = FN = 0
        [rows, cols] = self.gt_image.shape

        for i in range(rows):
            for j in range(cols):
                if self.gt_image[i, j] >= 1 and self.pred_image[i, j] >= 1:
                    TP = TP + 1
                if self.gt_image[i, j] == 0 and self.pred_image[i, j] == 0:
                    TN = TN + 1
                if self.gt_image[i, j] == 0 and self.pred_image[i, j] >= 1:
                    FP = FP + 1
                if self.gt_image[i, j] >= 1 and self.pred_image[i, j] == 0:
                    FN = FN + 1

        if (FP+TN) == 0:
            fpr = 0
        else:
            fpr = float(FP)/float(FP+TN)

        if (TP+FN) == 0:
            tpr = 0
        else:
            tpr = float(TP)/float(TP+FN)

        return TP, TN, FP, FN, fpr, tpr
    
    
    #Return precision   
    def precs(self):
    
       TP, TN, FP, FN, fpr, tpr = Evaluate.roc(self)
       try:
           prec = float(TP)/float(TP+FP)
       except ZeroDivisionError:
            prec = 0
       return prec

    #Return recall
    def rec(self):
        
       TP, TN, FP, FN, fpr, tpr = Evaluate.roc(self)

       try:
           rec = float(TP)/float(TP+FN)

       except ZeroDivisionError:
            rec = 0 
       return rec

    def f1score(self):            

        TP, TN, FP, FN, fpr, tpr = Evaluate.roc(self)
        try:
            f1score = 2*(float(prec * rec)/float(prec + rec))
        except ZeroDivisionError:
            f1score = 0
       
        with open('./evaluation_results/testdatasets_evaluation_metrics/linknet_roc_evaluationmetrics_test.txt', 'a')as f:
            print("Image num:", image_name, file =f)
            print('TP:', TP, file = f)
            print('TN:', TN, file = f)
            print('FP:', FP, file = f)
            print('FN:', FN, file = f)
            print("fpr:", fpr, file = f)
            print("tpr:", tpr, file = f)
            print("prec:", prec , file = f)
            print("rec:", rec, file = f)
            print("f1score:",f1score, file = f)
            print("--------------------", file = f)
            print("---------------------")
        
        header = ['image_name', 'TP', 'TN', 'FP', 'FN','FPR','TPR','Precion','Recall','F1score']
        
        # open the file in the write mode
        with open('./evaluation_results/testdatasets_evaluation_metrics/linknet_roc_evaluationmetrics_test.csv','a', newline = '') as f:
            # create the csv writer
            writer = csv.writer(f)
            data = [image_name, TP, TN, FP, FN,fpr, tpr, prec, rec, f1score]
            # write a row to the csv file
            writer.writerow(data)
            
    
        return f1score
    
# Main method.
if __name__ == "__main__":

    gt_image = './test_results/mask/'
    pred_image = './test_results/binary_map/'

    num_images = len(os.listdir(gt_image))
    num_images_pred = len(os.listdir(pred_image))
    print("num of gt image:", num_images)
    print("num of pred image:",num_images_pred)

    gt = [skimage.io.imread(file)
              for file in natsort.natsorted(glob.glob(gt_image + "*.jpg"))]

    pred = [skimage.io.imread(file)
            for file in natsort.natsorted(glob.glob(pred_image + "*.jpg"))]

    y_pred = []
    for i in range(len(pred)):
        #new_pred = color.rgb2gray(pred[i])
        new_pred = cv2.cvtColor(pred[i], cv2.COLOR_BGR2GRAY)
        y_pred.append(new_pred)
    '''
    y_gt = []
    for i in range(len(gt)):
        new_gt = cv2.cvtColor(gt[i], cv2.COLOR_BGR2GRAY)
        y_gt.append(new_gt)
    '''
    read_files = natsort.natsorted(glob.glob(gt_image + "*.jpg"))
    #print(read_files)
    
    roc_array = []
    prec_array = []
    rec_array = []
    f1score_array = []

    header = ['image_name', 'TP', 'TN', 'FP', 'FN','FPR','TPR','Precion','Recall','F1score']
        
        # open the file in the write mode
    with open('./evaluation_results/testdatasets_evaluation_metrics/linknet_roc_evaluationmetrics_test.csv','w', newline = '') as f:
            # create the csv writer
            writer = csv.writer(f)
            #data = [image_name, TP, TN, FP, FN,fpr, tpr, prec, rec, f1score]
            # write a row to the csv file
            writer.writerow(header)
            
    for i in range(num_images):

        image_name = read_files[i]
        
        val = Evaluate(y_pred[i], gt[i], image_name)

        print(image_name)
       
       
        roc = val.roc()
        
        prec = val.precs()
        
        rec = val.rec()
        
        f1score = val.f1score()
        
        
    
        
        roc_array.append(roc)
        prec_array.append(prec)
        rec_array.append(rec)
        f1score_array.append(f1score)
   
    
    
    prec = np.mean(np.array(prec_array))
    rec = np.mean(np.array(rec_array))
    f1score = np.mean(np.array(f1score_array))              

                   
    print(" ")
    
    print("Average Precision: ", str(round(prec, 3)))
    print("Average Recall: ", str(round(rec, 3)))
    print("Average F1-score: ", str(round(f1score, 3)))

    
    # Write results to file.
    f = open('./evaluation_results/testdatasets_evaluation_metrics/linknet_mean_roc_evaluation_metrics_test.txt', 'w')


    f.write('Average Precision: %s \n' % (str(round(prec, 3))))
    f.write('Average Recall: %s \n' % (str(round(rec, 3))))
    f.write('Average F1-score: %s \n' % (str(round(f1score, 3))))
    

    f.close()
    
