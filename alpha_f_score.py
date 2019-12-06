import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.svm import SVR
from sklearn.preprocessing import MinMaxScaler
from skfeature.function.statistical_based import f_score
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import LeaveOneOut

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

    loo = LeaveOneOut()
    loo.get_n_splits(X)

    for train_index, test_index in loo.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train = y[train_index]
        regressor.fit(X_train, y_train)
        y_pred.extend(regressor.predict(X_test))

    print(y_pred)

    # 6. count prediction accuracy
    y_true = y[0:]
    accuracy_score = r2_score(y_true, y_pred)
    rmse_score = mean_squared_error(y_true, y_pred)
    
    print(accuracy_score)
    print(rmse_score)

def average(value):
    return sum(value) / len(value)

if __name__ == "__main__":
    main()
    