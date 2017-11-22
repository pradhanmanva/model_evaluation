import sys
import warnings

from sklearn import svm
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, BaggingClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression, Perceptron
from sklearn.model_selection import KFold, cross_val_score, cross_val_predict
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier

from preprocessing import *

warnings.filterwarnings("ignore")


def get_classifiers():
    classifiers = [
        DecisionTreeClassifier(criterion="entropy"),
        Perceptron(penalty='l1'),
        MLPClassifier(solver='sgd', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1),
        MLPClassifier(solver='adam', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1),
        svm.LinearSVC(),
        GaussianNB(),
        LogisticRegression(random_state=0),
        KNeighborsClassifier(),
        BaggingClassifier(),
        RandomForestClassifier(max_depth=10, random_state=0),
        AdaBoostClassifier(random_state=0),
        GradientBoostingClassifier()
    ]
    return classifiers


if __name__ == '__main__':
    train_path, test_path = sys.argv[1:3]
    train_data, test_data = get_preprocessed_data(train_path, test_path)
    num_folds = sys.argv[3]
    models = get_classifiers()
    # ['dtc', 'perceptron', 'ann', 'dlp', 'svm', 'nb', 'logreg', 'knn', 'bagging', 'randomforest', 'adaboost','gradboost']
    features = train_data.columns.values[:-1]
    cls = train_data.columns.values[-1]
    X = train_data[features].values
    y = train_data[cls].values
    print("yes")
    kf = KFold(n_splits=2)
    kf.get_n_splits(X)  # returns the number of splitting iterations in the cross-validation
    print(kf)
    for train_index, test_index in kf.split(X):
        print('TRAIN:' + str(train_index) + 'TEST:' + str(test_index))
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        print(X_train)
        print(X_test)
        print(y_train)
        print(y_test)
        for c in models:
            print(c)
            classifier = c.fit(X_train, y_train)
            scores = cross_val_score(c, X_test, y_test, cv=6)
            print(scores)
            predictions = cross_val_predict(c, X_train, y_train, cv=6)
            print(predictions)
            # print(accuracy_score(y, predictions))
            # print(roc_curve(y, predictions))
            # print(precision_recall_fscore_support(y, predictions))
