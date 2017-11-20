from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier

from preprocessing import *


def decision_tree_classifier(df_train, df_test):
    '''kf = KFold(n_splits=2)
    print(kf.get_n_splits(X))
    for train_index,test_index in kf.split(X):'''
    features = ["s1", "c1", "s2", "c2", "s3", "c3", "s4", "c4", "s5", "c5"]
    cls = "class"

    inputs_train = df_train[features].values
    inputs_test = df_test[features].values
    classes_train = df_train[cls].values
    classes_test = df_test[cls].values

    dtc = DecisionTreeClassifier(criterion="entropy")
    dtc.fit(inputs_train, classes_train)
    print("DTC with entropy: " + str(dtc.score(inputs_test, classes_test) * 100))
    print("Cross Validation Average: " + str(cross_val_score(dtc, inputs_test, classes_test, cv=5).mean() * 100))

    dtc = DecisionTreeClassifier(criterion="gini")
    dtc.fit(inputs_train, classes_train)
    print("DTC with gini: " + str(dtc.score(inputs_test, classes_test) * 100))
    print("Cross Validation Average: " + str(cross_val_score(dtc, inputs_test, classes_test, cv=5).mean() * 100))


if __name__ == "__main__":
    d_train, d_test = import_data()
    decision_tree_classifier(d_train, d_test)
