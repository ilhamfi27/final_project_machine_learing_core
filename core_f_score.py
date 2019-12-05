import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.svm import SVR
from sklearn.preprocessing import MinMaxScaler
from skfeature.function.statistical_based import f_score
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error

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
    
    ranked_index = f_score.f_score(X, y, mode="index")
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
    # plt.title("R2 Score Plot")
    # plt.xlabel("Jumlah Fitur")
    # plt.ylabel("R2 Score")
    # plt.show()

    fig = plt.figure()
    fig.subplots_adjust(right=0.95, left=0.05)
    fig.suptitle("Hasil Testing Prediksi Fitur")
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

        # using leave one out cross validation
        for train_sequence in range(len(X)):
            X_train = []
            y_train = []
            predict_row = []
            for sequence in range(len(X)):
                if(train_sequence == sequence):
                    predict_row.append(X[sequence,0:repeat])
                else:
                    X_train.append(X[sequence,0:repeat])
                    y_train.append(y[sequence])
            regressor.fit(X_train, y_train)
            prediction_result = regressor.predict(predict_row)
            y_pred.extend(prediction_result)

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