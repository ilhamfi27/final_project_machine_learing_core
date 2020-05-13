import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.svm import SVR
from sklearn.preprocessing import MinMaxScaler
from skfeature.function.statistical_based import f_score
from skfeature.function.statistical_based import chi_square
from skfeature.function.statistical_based import CFS
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split


def main():
    sc = MinMaxScaler(feature_range=(0, 10))

    # 1. get data
    df = pd.read_excel('D:\\Private\\my_projects\\python\\final_project_core\\data_source\\e-commerce-dataset.xlsx')
    city_id = df.iloc[0:, 0].values
    raw_X = df.iloc[0:, 2:].values  # dataset
    raw_y = df.iloc[0:, 1].values  # label

    # 2. pre-processing
    clean_X = np.nan_to_num(raw_X)
    clean_y = np.nan_to_num(raw_y)

    # 3. normalization
    sc.fit(raw_X)
    X = np.array(sc.transform(clean_X))
    y = np.array(clean_y)

    # # 4. feature selection
    best_sort_feature = []
    ranked_index = f_score.f_score(X, y, mode="index")
    # print(ranked_index)
    # print()
    # raw_ranked_index = f_score.f_score(X, y, mode="raw")
    # for i in ranked_index:
    #     print(i, raw_ranked_index[i])

    # X_feature = X.astype(int)
    # y_label = y.astype(int)
    # ranked_index = chi_square.chi_square(X_feature, y_label, mode="index")

    # ranked_index = CFS.cfs(X, y)

    si_X = X[:, ranked_index[:]]

    kernel = ['rbf', 'linear']
    gamma = [1e-3, 1e-4]
    C = [1, 10, 100, 1000]
    epsilon = [0.1, 0.5, 1.0, 1.5, 2.0]

    loo = LeaveOneOut()
    loo.get_n_splits(si_X)

    total_search = (len(kernel) * len(gamma) * len(C) * len(epsilon)) - (len(C) * len(epsilon))
    have_search = 0
    best_search = []
    best_score = 0
    for k in kernel:
        for c in C:
            for e in epsilon:
                if k == 'rbf':
                    for g in gamma:
                        regressor = SVR(gamma=g, C=c, epsilon=e, kernel=k)

                        y_pred = []
                        y_true = []

                        for train_index, test_index in loo.split(si_X):
                            X_train, X_test = si_X[train_index], si_X[test_index]
                            y_train, y_test = y[train_index], y[test_index]
                            regressor.fit(X_train, y_train)
                            y_pred.extend(regressor.predict(X_test))
                            y_true.extend(y_test)

                        score = r2_score(y_true, y_pred)

                        if best_score < score:
                            best_score = score
                            best_search = [regressor, "gamma {} | c {} | epsilon {} | kernel {}".format(g, c, e, k)]

                        have_search += 1
                        print("=======================================================================================")
                        print("score", score)
                        print("gamma {} | c {} | epsilon {} | kernel {}".format(g, c, e, k))
                        print("progress", have_search, "out of", total_search)
                        print("=======================================================================================")
                else:
                    regressor = SVR(C=c, epsilon=e, kernel=k)

                    y_pred = []
                    y_true = []

                    for train_index, test_index in loo.split(si_X):
                        X_train, X_test = si_X[train_index], si_X[test_index]
                        y_train, y_test = y[train_index], y[test_index]
                        regressor.fit(X_train, y_train)
                        y_pred.extend(regressor.predict(X_test))
                        y_true.extend(y_test)

                    score = r2_score(y_true, y_pred)

                    if best_score < score:
                        best_score = score
                        best_search = [regressor, "c {} | epsilon {} | kernel {}".format(c, e, k)]

                    have_search += 1
                    print("=======================================================================================")
                    print("score", score)
                    print("c {} | epsilon {} | kernel {}".format(c, e, k))
                    print("progress", have_search, "out of", total_search)
                    print("=======================================================================================")

    print("best score", best_score)
    print("best search regressor", best_search[0])
    print("best search detail", best_search[1])


if __name__ == "__main__":
    main()
