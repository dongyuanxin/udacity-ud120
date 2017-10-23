import sys
import matplotlib.pyplot as plt
import copy
import numpy as np
import pylab as pl
try:
    from class_vis import prettyPicture
    from prep_terrain_data import makeTerrainData
except Exception as error:
    from .class_vis import prettyPicture
    from .prep_terrain_data import makeTerrainData

features_train, labels_train, features_test, labels_test = makeTerrainData()

from sklearn.svm import SVC
clf = SVC(kernel="rbf",gamma=1000)
# clf = SVC(kernel='rbf')
clf.fit(features_train,labels_train)

from sklearn.metrics import accuracy_score
pred = clf.predict(features_test)
right_rate = accuracy_score(y_pred=pred,y_true=labels_test)

print("The right rate is %.4f" % right_rate)
prettyPicture(clf, features_test, labels_test)
