import pickle
import numpy as np
from sklearn import  metrics
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.feature_selection import chi2, SelectKBest
from sklearn.naive_bayes import GaussianNB
from datetime import datetime
import matplotlib.pyplot as plt
from load_data import unpickle, getCIFAR10


if __name__ == '__main__':
    startTime = datetime.now()
    direc = 'data/'
    test_file = 'test_batch'
    filename = 'data_batch_'
    X_train, y_train = getCIFAR10(direc, filename, 5)
    X = SelectKBest(chi2, k=2500).fit_transform(X_train, y_train)

    data_test = unpickle(direc + test_file)
    X_test = data_test[b'data']
    y_test = data_test[b'labels']

    print("Computation under process")
    print("Please Wait...")

    clf1 = LinearDiscriminantAnalysis()
    clf2 = RandomForestClassifier(n_estimators=50, random_state=1)
    clf3 = GaussianNB()
    eclf1 = VotingClassifier(estimators=[
        ('lr', clf1), ('rf', clf2), ('gnb', clf3)], voting='soft')
    eclf2 = eclf1.fit(X_train, y_train)
    print(eclf2.predict(X_test))
    # Yte_predictrbf = clf.predict(X_test)
    print("Prediction complete")
    print('The accuracy of classifier on test data: {:.2f}'.format((eclf2.score(X_test, y_test) * 100)))
    print(datetime.now() - startTime)
    metrics.plot_roc_curve(eclf2, X_test, y_test)  # doctest: +SKIP
    plt.show()
