import matplotlib.pyplot as plt
import pandas as pd
import seaborn


def print_plots(df):
    for attr in list(df.columns.values):
        df[attr].plot.hist()
        print(plt.show())
    seaborn.pairplot(df, hue='class')
    print(plt.show())


def import_data(training_data, test_data):
    df_train = pd.read_csv(training_data, header=None)
    df_test = pd.read_csv(test_data, header=None)
    df_train.columns = ['s1', 'c1', 's2', 'c2', 's3', 'c3', 's4', 'c4', 's5', 'c5', 'class']
    df_test.columns = ['s1', 'c1', 's2', 'c2', 's3', 'c3', 's4', 'c4', 's5', 'c5', 'class']
    print('Unprocessed Training Shape: ' + str(df_train.shape))
    print('Unprocessed Testing Shape: ' + str(df_test.shape))
    df_train = df_train.drop_duplicates()
    df_test = df_test.drop_duplicates()
    df_train = df_train.dropna()
    df_test = df_test.dropna()
    print('Preprocessed Training Shape: ' + str(df_train.shape))
    print('Preprocessed Testing Shape: ' + str(df_test.shape))
    return df_train, df_test


def get_preprocessed_data(path_train, path_test):
    train, test = import_data(path_train, path_test)
    print(pd.DataFrame.transpose(train.describe()))
    print(pd.DataFrame.transpose(test.describe()))
    # print_plots(train)
    # print_plots(test)
    print('Pre-processing done.')
    return train,test
# get_preprocessed_data('poker-train.csv', 'poker-test.csv', 'clean-train', 'clean-test')
