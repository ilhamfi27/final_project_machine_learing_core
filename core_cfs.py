import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.svm import SVR
from sklearn.preprocessing import MinMaxScaler
from skfeature.function.statistical_based import CFS
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import LeaveOneOut
from ten_feature_train import trainf

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

    # 4. feature selection
    best_sort_feature = []
    
    ranked_index = CFS.cfs(X, y)
    for row in X:
        row_array = []
        for num, feature_idx in enumerate(ranked_index):
            row_array.append(row[feature_idx])
        best_sort_feature.append(row_array)
    
    # 5. get best feature predict score
    best_pred, best_score, result, ten_column_predictions \
         = trainf(best_sort_feature, y)

    for num, score_res in enumerate(result):
        print("{}. {} column, R2_SCOREnya adalah {} dan RMSEnya adalah {}"
            .format(num + 1, score_res[0], score_res[1], score_res[2]))

    result = np.array(result)
    plt.scatter(result[0:, 0], result[0:, 1])
    plt.plot(result[0:, 0], result[0:, 1])
    plt.title("Penilaian Akurasi Dengan R2 Score (CFS)")
    plt.xlabel("Jumlah Fitur")
    plt.ylabel("R2 Score")
    plt.show()

    fig = plt.figure()
    fig.subplots_adjust(hspace=0.2, wspace=0.15, bottom=0.05, right=0.95, left=0.05)
    fig.suptitle("Hasil Prediksi Fitur Dengan CFS")
    for i, data in enumerate(ten_column_predictions):
        ax = fig.add_subplot(2, 5, (i + 1))
        ax.scatter(y, data)
        ax.plot(y, y)
        ax.set_title("{} Fitur".format(result[i][0]))
    plt.show()

    plt.scatter(y, best_pred)
    plt.plot(y, y)
    plt.title("Hasil Prediksi Terbaik (CFS)")
    plt.xlabel("Data Real")
    plt.ylabel("Data Prediksi")
    plt.show()

if __name__ == "__main__":
    main()
