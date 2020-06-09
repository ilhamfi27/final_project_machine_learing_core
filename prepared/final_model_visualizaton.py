import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.svm import SVR
from sklearn.preprocessing import MinMaxScaler
import getopt, sys
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
import tkinter
from tkinter import filedialog


root = tkinter.Tk()
root.withdraw()

def main():
    import_file_path = filedialog.askopenfilename()
    sc = MinMaxScaler(feature_range=(0, 10))

    # 1. get data
    df = pd.read_excel(import_file_path)
    city_id = np.asarray(df['city_id'])
    raw_X = np.asarray(df.loc[:, 'sum_price_car':'std_buyer_land_rent']) # features
    raw_y = np.asarray(df['BPS_poverty_rate'])  # label

    # 2. pre-processing
    clean_X = np.nan_to_num(raw_X)
    clean_y = np.nan_to_num(raw_y)

    # 3. normalization
    sc.fit(raw_X)
    X = np.array(sc.transform(clean_X))
    y = np.array(clean_y)

    ranked_index = [
        49, 60, 62, 63, 37, 13, 14, 61, 38, 50,
        69, 84, 66, 86, 48, 85, 12, 36, 0, 1,
        57, 54, 3, 76, 51, 75, 2, 77, 6, 64,
        9, 15, 56, 18, 22, 82, 83, 21, 24, 87,
        10, 39, 42, 70, 45, 79, 71, 25, 20, 67,
        72, 68, 23, 94, 91, 34, 8, 59, 92, 65,
        11, 80, 27, 35, 93, 89, 26, 31, 95, 44,
        32, 88, 47, 55, 90, 52, 16, 17, 30, 73,
        7, 4, 53, 33, 43, 46, 5, 19, 41, 28
    ]
    # SVR
    regressor = SVR(gamma=0.001, C=10, epsilon=0.5, kernel='rbf')

    # 5. get best feature predict score
    best_pred, best_score, result, ten_column_predictions, best_regressor, best_features \
        = trainf(X[:, ranked_index[:]], y, regressor)

    for i, x in enumerate(best_pred):
        if abs(y[i] - x) > 1.5:
            plt.scatter(y[i], x, c="r", s=15)
        else:
            plt.scatter(y[i], x, c="b", s=15)
    plt.plot(y, y)
    plt.plot(y - 1.5, y, c="y", linewidth="0.5")
    plt.plot(y + 1.5, y, c="y", linewidth="0.5")
    plt.xlabel("Actual Data")
    plt.ylabel("Prediction Data")
    plt.show()

def trainf(X, y, regressor):
    repeat = 0
    X = np.array(X)
    X_column = X.shape[1]
    result = []
    best_score = -999999
    best_pred = []
    ten_column_predictions = []
    best_regressor = None
    best_features = []

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
            best_regressor = regressor
            best_features = repeat

    return best_pred, best_score, result, ten_column_predictions, best_regressor, best_features

if __name__ == "__main__":
    main()
