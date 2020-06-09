import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tkinter
from tkinter import filedialog
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

root = tkinter.Tk()
root.withdraw()


def main():
    import_file_path = filedialog.askopenfilename()
    sc = MinMaxScaler(feature_range=(0, 10))

    # 1. get data
    df = pd.read_excel(import_file_path)
    raw_X = np.asarray(df.loc[:, 'sum_price_car':'std_buyer_land_rent'])  # features
    raw_y = np.asarray(df['BPS_poverty_rate'])  # label

    # 2. pre-processing
    clean_X = np.nan_to_num(raw_X)
    clean_y = np.nan_to_num(raw_y)

    # 3. normalization
    sc.fit(raw_X)
    X = np.array(sc.transform(clean_X))
    y = np.array(clean_y)

    # # 4. feature selection
    best_sort_feature = []
    f_score_ranked_index = f_score.f_score(X, y, mode="index")

    X_feature = X.astype(int)
    y_label = y.astype(int)
    chi_square_ranked_index = chi_square.chi_square(X_feature, y_label, mode="index")

    # ranked_index = CFS.cfs(X, y)

    kernel = ['rbf']
    gamma = [1e-3, 1e-4]
    C = [1, 10, 100, 1000]
    epsilon = [0.1, 0.5, 1.0, 1.5, 2.0]
    fs_algorithms = [f_score_ranked_index, chi_square_ranked_index]

    total_search = (len(kernel) * len(gamma) * len(C) * len(epsilon) * len(fs_algorithms))
    have_search = 0
    best_search = []
    overall_best_score = 0
    tuning_record = []


    for ranked_index in fs_algorithms:
        si_X = X[:, ranked_index[:]]

        loo = LeaveOneOut()
        loo.get_n_splits(si_X)
        for k in kernel:
            for g in gamma:
                for c in C:
                    for e in epsilon:
                        regressor = SVR(gamma=g, C=c, epsilon=e, kernel=k)
                        best_pred, best_score, result, ten_column_predictions, total_best_features = trainf(si_X, y,
                                                                                                      regressor)
                        score = best_score

                        if overall_best_score < score:
                            overall_best_score = score
                            best_search = [
                                regressor,
                                "gamma {} | c {} | epsilon {} | kernel {} | total features {}"
                                    .format(g, c,e, k,total_best_features),
                                ranked_index[:total_best_features]
                           ]

                        have_search += 1
                        print("=======================================================================================")
                        print("score", score)
                        print("gamma {} | c {} | epsilon {} | kernel {} | total features {}"
                                .format(g, c, e, k, total_best_features))
                        print("progress", have_search, "out of", total_search)
                        print("=======================================================================================")
                        tuning_record.append([
                            g, c, e, k,
                            total_best_features, score, ",".join(str(s) for s in ranked_index[:total_best_features])
                        ])

    print("best score", overall_best_score)
    print("best search regressor", best_search[0])
    print("best search detail", best_search[1])
    print("selected FS", best_search[2])

    df1 = pd.DataFrame(tuning_record,
                       columns=['gamma', 'C', 'epsilon', 'kernel', 'total of used features', 'r2 score', 'used features'])
    df1.to_excel("tuning record.xlsx", index=False,)


def trainf(X, y, regressor):
    repeat = 0
    X = np.array(X)
    X_column = X.shape[1]
    result = []
    best_score = -999999
    best_pred = []
    ten_column_predictions = []
    total_best_features = 0

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
            total_best_features = repeat

    return best_pred, best_score, result, ten_column_predictions, total_best_features


if __name__ == "__main__":
    main()
