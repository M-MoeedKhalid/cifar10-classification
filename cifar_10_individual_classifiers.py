import pickle
import numpy as np
from sklearn import svm
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from load_data import unpickle, getCIFAR10

if __name__ == '__main__':
    direc = 'data/'
    test_file = 'test_batch'
    filename = 'data_batch_'
    X_train, y_train = getCIFAR10(direc, filename, 5)
    data_test = unpickle(direc + test_file)
    X_test = data_test[b'data']
    y_test = data_test[b'labels']
    # Call kNN
    # k = int(input("Enter the value of k for k-Nearest Neighbor Classifier: "))
    print("Computation under process")
    print("Please Wait...")
    gnb = GaussianNB()
    gnb.fit(X_train, y_train)
    Yte_predictgnb = gnb.predict(X_test)
    print("Prediction complete")
    print('The accuracy of classifier on test data: {:.2f}'.format((gnb.score(X_test, y_test) * 100)))

    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(X_train, y_train)
    Yte_predictknn = knn.predict(X_test)
    print("Prediction complete")
    print('The accuracy of classifier on test data: {:.2f}'.format((knn.score(X_test, y_test) * 100)))

    rbf = svm.SVC(
        # C=1.0,
        kernel='rbf',
        degree=3
        ,
        gamma=2, max_iter=1e5
    )

    rbf.fit(X_train, y_train)
    Yte_predictrbf = rbf.predict(X_test)
    print("Prediction complete")
    print('The accuracy of classifier on test data: {:.2f}'.format((rbf.score(X_test, y_test) * 100)))
