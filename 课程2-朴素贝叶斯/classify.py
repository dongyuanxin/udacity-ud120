from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
import numpy as np

def NBAccuracy(features_train, labels_train, features_test, labels_test):
    clf = GaussianNB()
    clf.fit(features_train,labels_train)
    pred = clf.predict(features_test) # 模型预测的结果

    accuracy = accuracy_score(y_true=labels_test,y_pred=pred) # 得到准确率
    return accuracy