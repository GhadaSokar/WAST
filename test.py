from sklearn import svm

# train SVM classifier with the selected features
def svm_test(train_X_new, train_y, test_X, test_y, selected_features):
    clf = svm.SVC()
    clf.fit(train_X_new, train_y)
    SVCacc = float(clf.score(test_X[:, selected_features], test_y))
    return SVCacc
