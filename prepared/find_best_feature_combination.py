import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.svm import SVR
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from skfeature.function.statistical_based import f_score, chi_square, CFS
import tkinter
from tkinter import filedialog
import getopt, sys


root = tkinter.Tk()
root.withdraw()

full_arguments = sys.argv
argument_list = full_arguments[1:]

unix_options = "s:"
gnu_options = ["selection="]

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

    f_score_index = f_score.f_score(X, y, mode="index")

    X_feature = X.astype(int)
    y_label = y.astype(int)
    chi_square_index = chi_square.chi_square(X_feature, y_label, mode="index")

    cfs_index = CFS.cfs(X, y)
    # SVR
    regressor = SVR(gamma=0.001, C=10, epsilon=0.5, kernel='rbf')
    try:
        ranked_index = []
        title = ""
        arguments, values = getopt.getopt(argument_list, unix_options, gnu_options)
        for current_argument, current_value in arguments:
            if current_argument in ("-s", "--selection"):
                if current_value == "f_score":
                    ranked_index = f_score_index
                    title = "F-Score Feature Experiment"
                elif current_value == "chi_square":
                    ranked_index = chi_square_index
                    title = "Chi-square Feature Experiment"
                elif current_value == "cfs":
                    ranked_index = cfs_index
                    title = "CFS Feature Experiment"
                else:
                    print("Invalid option value")
            else:
                print("Invalid option value")

        if len(ranked_index) > 0:
            # 5. get best feature predict score
            best_pred, best_score, result, ten_column_predictions, best_regressor, best_features \
                = trainf(X[:, ranked_index[:]], y, regressor)

            fig = plt.figure()
            fig.suptitle(title)
            fig.subplots_adjust(hspace=0.35, wspace=0.28, top=0.9, bottom=0.1, right=0.95, left=0.05)
            for i, data in enumerate(ten_column_predictions):
                ax = fig.add_subplot(2, 5, (i + 1))
                for _i, x in enumerate(data):
                    if abs(y[_i] - x) > 1.5:
                        ax.scatter(y[_i], x, c="r", s=5)
                    else:
                        ax.scatter(y[_i], x, c="b", s=5)
                # ax.scatter(y, data)
                ax.plot(y - 1.5, y, c="y", linewidth="0.5")
                ax.plot(y + 1.5, y, c="y", linewidth="0.5")
                ax.plot(y, y, linewidth="0.5")
                ax.set_title("{} Feature".format(result[i][0]))
                ax.set_xlabel("Actual Data")
                ax.set_ylabel("Prediction Data")
            plt.show()
    except getopt.error as err:
        print(str(err))

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
