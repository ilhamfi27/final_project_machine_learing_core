import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.svm import SVR
from sklearn.preprocessing import MinMaxScaler
from skfeature.function.statistical_based import chi_square
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import LeaveOneOut

def main():
    sc = MinMaxScaler(feature_range=(0,10))

    # 1. get data
    df = pd.read_excel('e-commerce-dataset.xlsx')
    city_id = df.iloc[0:, 0].values
    raw_X = df.iloc[0:,2:].values # dataset
    raw_y = df.iloc[0:,1].values # label

    # 2. pre-processing
    clean_X = np.nan_to_num(raw_X)
    clean_y = np.nan_to_num(raw_y)

    # 3. normalization
    sc.fit(raw_X)
    X = np.array(sc.transform(clean_X))
    y = np.array(clean_y)

    # # 4. feature selection
    best_sort_feature = []
    
    X_feature = X.astype(int)
    y_label = y.astype(int)
    ranked_index = chi_square.chi_square(X_feature, y_label, mode="index")
    for row in X:
        row_array = []
        for num, feature_idx in enumerate(ranked_index):
            row_array.append(row[feature_idx])
        best_sort_feature.append(row_array)
    
    # 5. get best feature predict score
    best_pred, best_score, result, ten_column_predictions \
         = train_per_10_feature(best_sort_feature, y)

    # result = np.array(result)
    # plt.scatter(result[0:, 0], result[0:, 1])
    # plt.plot(result[0:, 0], result[0:, 1])
    # plt.title("Chi Score Plot")
    # plt.xlabel("Jumlah Fitur")
    # plt.ylabel("Chi Score")
    # plt.show()

    fig = plt.figure()
    fig.subplots_adjust(hspace=0.2, wspace=0.15, bottom=0.05, right=0.95, left=0.05)
    fig.suptitle("Hasil Prediksi Fitur Dengan Chi Square")
    for i, data in enumerate(ten_column_predictions):
        ax = fig.add_subplot(2, 5, (i + 1))
        ax.scatter(y, data)
        ax.plot(y, y)
        ax.set_title("{} Fitur".format(result[i][0]))
    plt.show()

def train_per_10_feature(X, y):
    repeat = 0
    X = np.array(X)
    X_column = X.shape[1]
    result = []
    best_score = 0
    best_pred = []
    ten_column_predictions = []

    while repeat < X_column - 1:
        score = []
        if (X_column - repeat) < 10 :
            repeat += (X_column - repeat)
        else:
            repeat += 10
    
        # predict
        regressor = SVR(gamma='scale', C=1.0, epsilon=0.2)
        y_pred = []
        
        X_selected = X[0:, 0:repeat]
        loo = LeaveOneOut()
        loo.get_n_splits(X)

        for train_index, test_index in loo.split(X_selected):
            X_train, X_test = X_selected[train_index], X_selected[test_index]
            y_train = y[train_index]
            regressor.fit(X_train, y_train)
            y_pred.extend(regressor.predict(X_test))

        # count accuracy prediction
        y_true = y[0:]
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
