# coding:utf-8
from sklearn import preprocessing
from sklearn.metrics import accuracy_score, recall_score, precision_score
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import numpy as np
y_test = np.array(['7', '1', '2', '1'])
preds = np.array(['7', '2', '2', '2'])
# print(accuracy_score(y_test, preds))  # 2/4
# print(recall_score(y_test, preds, pos_label=1))  # 默认查看1的召回率2/2
# print(precision_score(y_test, preds, pos_label=1))  # 默认查看1的准确率2/4
print(classification_report(y_test, preds, target_names=['class 0', 'class 1', 'class 2']))
# tn, fp, fn, tp = confusion_matrix(y_test, preds, labels=[0, 1])


le = preprocessing.LabelEncoder()
le.fit([4, 12, 24, 3, 7, 30, 90, 180])
# [  3   4   7  12  24  30  90 180]
print(le.classes_)
print(le.transform([4, 12, 24, 3, 7, 30, 90, 180]))
# 1, 3, 4, 0, 2, 5, 6, 7
