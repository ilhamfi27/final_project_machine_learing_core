import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.svm import SVR
from sklearn.preprocessing import MinMaxScaler
from skfeature.function.statistical_based import f_score
from sklearn.metrics import r2_score

def main():
    sc = MinMaxScaler()

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
    selected_feature = []

    fs_score = f_score.f_score(X, y, mode="raw")
    avg_score = average(fs_score)

    for row in range(len(X)):
        row_array = []
        feature_order = 0
        for score in fs_score:
            if score > avg_score:
                row_array.append(X[row][feature_order])
            feature_order += 1
        selected_feature.append(row_array)

    # 5. predict
    regressor = SVR(gamma='scale', C=1.0, epsilon=0.2)

    selected_feature = np.array(selected_feature)
    X_train = selected_feature[0:59,1:]
    y_train = y[0:59]

    regressor.fit(X_train, y_train)

    y_pred = regressor.predict(selected_feature[59:118, 1:])
    print(y_pred)

    # 6. count accuracy prediction
    y_true = y[59:118]
    accuracy_score = r2_score(y_true, y_pred)
    
    print(accuracy_score)

def average(value):
    return sum(value) / len(value)

if __name__ == "__main__":
    main()