from sklearn.tree import DecisionTreeClassifier

def classify(features_train, labels_train):
    ### your code goes here--should return a trained decision tree classifer
    # print(len(features_train),len(labels_train))
    clf = DecisionTreeClassifier(criterion="entropy",min_samples_split=50).fit(features_train,labels_train)

    return clf