# coding:utf-8
from sklearn.metrics import accuracy_score, recall_score, precision_score
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import numpy as np
y_test = np.array([1, 0, 1, 0])
preds = np.array([1, 1, 1, 1])
print(accuracy_score(y_test, preds))  # 2/4
print(recall_score(y_test, preds, pos_label=1))  # 默认查看1的召回率2/2
print(precision_score(y_test, preds, pos_label=1))  # 默认查看1的准确率2/4
print(classification_report(y_test, preds, target_names=['class 0', 'class 1']))
# tn, fp, fn, tp = confusion_matrix(y_test, preds, labels=[0, 1])
