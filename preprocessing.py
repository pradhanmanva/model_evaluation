import matplotlib.pyplot as plt
import pandas as pd
import seaborn


def check_data(df):
    print(df.dtypes)
    print(df.describe())
    print(df.isnull().any())


def print_plots(df):
    df['s1'].plot.hist()
    print(plt.show())
    df['s2'].plot.hist()
    print(plt.show())
    df['s3'].plot.hist()
    print(plt.show())
    df['s4'].plot.hist()
    print(plt.show())
    df['s5'].plot.hist()
    print(plt.show())
    df['c1'].plot.hist()
    print(plt.show())
    df['c2'].plot.hist()
    print(plt.show())
    df['c3'].plot.hist()
    print(plt.show())
    df['c4'].plot.hist()
    print(plt.show())
    df['c5'].plot.hist()
    print(plt.show())
    seaborn.pairplot(df, hue='class')
    print(plt.show())


def import_data(training_data, test_data):
    df_train = pd.read_csv(training_data, header=None)
    df_test = pd.read_csv(test_data, header=None)
    df_train.columns = ["s1", "c1", "s2", "c2", "s3", "c3", "s4", "c4", "s5", "c5", "class"]
    df_test.columns = ["s1", "c1", "s2", "c2", "s3", "c3", "s4", "c4", "s5", "c5", "class"]
    print(df_train.shape)
    print(df_test.shape)
    df_train = df_train.drop_duplicates()
    df_test = df_test.drop_duplicates()
    df_train = df_train.dropna()
    df_test = df_test.dropna()
    print(df_train.shape)
    print(df_test.shape)
    return df_train, df_test


def get_preprecessed_data(path_train, path_test):
    train, test = import_data(path_train, path_test)
    check_data(train)
    check_data(test)
    # print_plots(train)
    # print_plots(test)
    pd.DataFrame.to_csv(train, "clean_train.csv")
    pd.DataFrame.to_csv(test, "clean_test.csv")

get_preprecessed_data("")