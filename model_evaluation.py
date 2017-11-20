'''
numFolds = k #(where k >= 10)
classifiers = {c1, c2, …, cn} #list of n classifiers with best parameters
split the data into k folds d[1…k]
for i in 1 to numFolds
    # create training dataset by combining all folds except d[i]
    train = {d[1] + d[2] + … + d[i-1] + d[i+1] + … + d[k]}
    # create test dataset using d[i]
    test = d[i]
    for c in classifiers classifiers
        # create a model of type c using train
        model <- createModel(c, train)
        # find accuracy of model of type c on test
        for classifier c: accuracy[i] <- findAccuracy(model, test)
        for classifier c: other_parameter[i] <- findEvaluation(model, test)
At the end of the code, you will output the average of the accuracy and
other evaluation parameter for each classifier i.e.
average accuracy = average (accuracy[1], accuracy[2], …, accuracy[n])
average other_parameter = average (other_parameter[1], other_parameter[2], …,other_parameter [n])
'''


if __name__ == "__main__":

    num_folds = 5
    classifiers = get_classifiers()