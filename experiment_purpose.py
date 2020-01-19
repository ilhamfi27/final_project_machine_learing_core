import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.svm import SVR
from sklearn.preprocessing import MinMaxScaler
from skfeature.function.statistical_based import f_score
from skfeature.function.statistical_based import chi_square
from skfeature.function.statistical_based import CFS
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import LeaveOneOut
from ten_feature_train import trainf
import getopt, sys
from csv_push import CSVPush

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
    
    # ranked_index = f_score.f_score(X, y, mode="index")
    
    X_feature = X.astype(int)
    y_label = y.astype(int)
    ranked_index = chi_square.chi_square(X_feature, y_label, mode="index")

    # ranked_index = CFS.cfs(X, y)

    for row in X:
        row_array = []
        for num, feature_idx in enumerate(ranked_index):
            row_array.append(row[feature_idx])
        best_sort_feature.append(row_array)
    
    # 5. get best feature predict score
    best_pred, best_score, result, ten_column_predictions \
         = trainf(best_sort_feature, y)

    full_arguments = sys.argv
    argument_list = full_arguments[1:]

    unix_options = "hs:f:"
    gnu_options = ["help", "show="]

    try:
        arguments, values = getopt.getopt(argument_list, unix_options, gnu_options)
        for current_argument, current_value in arguments:
            if current_argument in ("-h", "--help"):
                print("\n-h or --help\t\tFor displaying help\n")
                print("use option -s or --show= for showing specific result\n")
                print("show options:")
                print("-s scoreDetail or --show=scoreDetail")
                print("-s featureGraph or --show=featureGraph")
                print("-s tenGraph or --show=tenGraph")
                print("-s bestPrediction or --show=bestPrediction")
            elif current_argument in ("-s", "--show"):
                if current_value == "scoreDetail":
                    for num, score_res in enumerate(result):
                        print("{}. {} column, R2_SCOREnya adalah {} dan RMSEnya adalah {}"
                            .format(num + 1, score_res[0], score_res[1], score_res[2]))
                elif  current_value == "featureGraph":
                    result = np.array(result)
                    plt.scatter(result[0:, 0], result[0:, 2])
                    plt.plot(result[0:, 0], result[0:, 2])
                    for i, txt in enumerate(result[0:, 2]):
                        plt.annotate("%.3f" % txt, (result[0:, 0][i], result[0:, 2][i]),
                                xycoords='data',
                                xytext=(15, 25), textcoords='offset points',
                                bbox=dict(boxstyle='round, pad=0.2', fc='yellow', alpha=0.7),
                                horizontalalignment='right', verticalalignment='top',
                                arrowprops=dict(
                                    arrowstyle='->', color='black'
                                )
                            )
                    plt.xlabel("n-Feature")
                    # plt.ylabel("R\u00b2 Score") # R<sup>2</sup> Score
                    plt.ylabel("RMSE Score")
                    plt.show()
                elif  current_value == "tenGraph":
                    fig = plt.figure()
                    fig.subplots_adjust(hspace=0.2, wspace=0.15, bottom=0.05, right=0.95, left=0.05)
                    for i, data in enumerate(ten_column_predictions):
                        ax = fig.add_subplot(2, 5, (i + 1))
                        ax.scatter(y, data)
                        ax.plot(y, y)
                        ax.set_title("{} Feature".format(result[i][0]))
                    plt.show()
                elif  current_value == "bestPrediction":
                    plt.scatter(y, best_pred)
                    plt.plot(y, y)
                    plt.xlabel("Actual Data")
                    plt.ylabel("Prediction Data")
                    plt.show()
                else:
                    print("Invalid option value")
    except getopt.error as err:
        print(str(err))

if __name__ == "__main__":
    main()
