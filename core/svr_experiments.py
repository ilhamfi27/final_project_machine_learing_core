import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import MinMaxScaler
from skfeature.function.statistical_based import f_score
from skfeature.function.statistical_based import chi_square
from skfeature.function.statistical_based import CFS
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import LeaveOneOut


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

    best_sort_feature = []
    # # 4. feature selection
    ranked_index = f_score.f_score(X, y, mode="index")
    print("F SCORE")
    print(ranked_index)

    # print("\n=============================================================\n")
    # X_feature = X.astype(int)
    # y_label = y.astype(int)
    # ranked_index = chi_square.chi_square(X_feature, y_label, mode="index")
    # print("CHI SQUARE")
    # print(ranked_index)
    #
    # print("\n=============================================================\n")
    # ranked_index = CFS.cfs(X, y)
    # print("CFS")
    # print(ranked_index)

    # SVR
    regressor = SVR(gamma=0.0001, C=100, epsilon=0.1, kernel='rbf')
    # get best feature predict score
    best_pred, best_score, result, ten_column_predictions \
        = trainf(X[:, ranked_index[:]], y, regressor=regressor)

    print("SVR")
    for num, score_res in enumerate(result):
        print("{}. {} column, R2_SCOREnya adalah {} dan RMSEnya adalah {}"
              .format(num + 1, score_res[0], score_res[1], score_res[2]))
    print("\n==============================================================\n")

    # Linear Regression
    regressor = LinearRegression()
    # get best feature predict score
    best_pred, best_score, result, ten_column_predictions \
        = trainf(X[:, ranked_index[:]], y, regressor=regressor)

    print("Linear Regression")
    for num, score_res in enumerate(result):
        print("{}. {} column, R2_SCOREnya adalah {} dan RMSEnya adalah {}"
              .format(num + 1, score_res[0], score_res[1], score_res[2]))
    print("\n==============================================================\n")

    # KNN Regressor
    regressor = KNeighborsRegressor(n_neighbors=1)
    # get best feature predict score
    best_pred, best_score, result, ten_column_predictions \
        = trainf(X[:, ranked_index[:]], y, regressor=regressor)

    print("KNN Regressor")
    for num, score_res in enumerate(result):
        print("{}. {} column, R2_SCOREnya adalah {} dan RMSEnya adalah {}"
              .format(num + 1, score_res[0], score_res[1], score_res[2]))
    print("\n==============================================================\n")

    # predict
    # regressor = SVR(gamma=0.0001, C=100, epsilon=0.1, kernel='rbf')
    # y_pred = []
    # y_true = []
    #
    # loo = LeaveOneOut()
    # loo.get_n_splits(X)
    #
    # si_X = best_features[:, :96]
    # X_selected = si_X
    # loo = LeaveOneOut()
    # loo.get_n_splits(si_X)
    #
    # for train_index, test_index in loo.split(X_selected):
    #     X_train, X_test = X_selected[train_index], X_selected[test_index]
    #     y_train, y_test = y[train_index], y[test_index]
    #
    #     regressor.fit(X_train, y_train)
    #     y_pred.extend(regressor.predict(X_test))
    #     y_true.extend(y_test)
    #
    # print(r2_score(y_true, y_pred))

def trainf(X, y, regressor):
    repeat = 0
    X = np.array(X)
    X_column = X.shape[1]
    result = []
    best_score = 0
    best_pred = []
    ten_column_predictions = []

    while repeat < X_column - 1:
        score = []
        if (X_column - repeat) < 10:
            repeat += (X_column - repeat)
        else:
            repeat += 10

        # predict
        y_pred = []
        y_true = []

        X_selected = X[0:, 0:repeat]
        loo = LeaveOneOut()
        loo.get_n_splits(X)

        for train_index, test_index in loo.split(X_selected):
            X_train, X_test = X_selected[train_index], X_selected[test_index]
            y_train, y_test = y[train_index], y[test_index]
            regressor.fit(X_train, y_train)
            y_pred.extend(regressor.predict(X_test))
            y_true.extend(y_test)

        # count accuracy prediction
        accuracy_score = r2_score(y_true, y_pred)
        rmse_score = mean_squared_error(y_true, y_pred)

        score.append(repeat)
        score.append(accuracy_score)
        score.append(rmse_score)

        result.append(score)

        ten_column_predictions.append(y_pred)

        if best_score < accuracy_score:
            best_pred = y_pred
            best_score = accuracy_score

    return best_pred, best_score, result, ten_column_predictions

if __name__ == "__main__":
    main()