import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.svm import SVR
from sklearn.preprocessing import MinMaxScaler
from skfeature.function.statistical_based import f_score
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import train_test_split

def main():
    sc = MinMaxScaler(feature_range=(0,10))

    # 1. get data
    df = pd.read_excel('e-commerce-dataset.xlsx')
    raw_X = df.iloc[0:,2:].values # dataset
    raw_y = df.iloc[0:,1].values # label

    # 2. pre-processing
    clean_X = np.nan_to_num(raw_X)
    clean_y = np.nan_to_num(raw_y)

    # 3. normalization
    sc.fit(raw_X)
    X = sc.transform(clean_X)
    y = clean_y

    # 4. feature selection
    ranked_index = []

    fs_score = f_score.f_score(X, y, mode="index")

    for row in X:
        row_array = []
        for num, feature_idx in enumerate(fs_score):
            row_array.append(row[feature_idx])
        ranked_index.append(row_array)

    # 5. predict
    regressor = SVR(gamma='scale', C=1.0, epsilon=0.2)
    X = np.array(ranked_index)
    y_pred = []
    y_true = []

    # X_train, X_test, y_train, y_test = \
    #     train_test_split(X, y, test_size=0.5, shuffle=False)
    # regressor.fit(X_train, y_train)
    # y_pred = regressor.predict(X_test)
    # y_true = y_test

    loo = LeaveOneOut()
    loo.get_n_splits(X)

    for train_index, test_index in loo.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        regressor.fit(X_train, y_train)
        y_pred.extend(regressor.predict(X_test))
        y_true.extend(y_test)

    # print(y_pred)

    # 6. count prediction accuracy
    accuracy_score  = r2_score(y_true, y_pred)
    rmse_score = mean_squared_error(y_true, y_pred)
    
    print("Skor Akurasi : ", accuracy_score)
    print("Skor RMSE : ", rmse_score)

    # plotting
    plt.scatter(y_true, y_pred)
    plt.plot(y_true, y_true)
    plt.title("Plot Hasil Prediksi SVR")
    plt.xlabel("Hasil Prediksi")
    plt.ylabel("Label Awal")
    plt.show()

def average(value):
    return sum(value) / len(value)

if __name__ == "__main__":
    main()
    