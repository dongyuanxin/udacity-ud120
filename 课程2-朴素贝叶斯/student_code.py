
import matplotlib.pyplot as plt
import numpy as np
import pylab as pl

try:
    # from class_vis import prettyPicture
    from prep_terrain_data import makeTerrainData
    from classify import NBAccuracy
except Exception as e:
    # from .class_vis import prettyPicture
    from .prep_terrain_data import makeTerrainData
    from .classify import NBAccuracy
features_train, labels_train, features_test, labels_test = makeTerrainData()

def submitAccuracy():
    accuracy = NBAccuracy(features_train, labels_train, features_test, labels_test)
    return accuracy

if __name__=='__main__':
    print("这次的准确率：",submitAccuracy())