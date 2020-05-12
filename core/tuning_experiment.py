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
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split


def main():
    sc = MinMaxScaler(feature_range=(0, 10))

    # 1. get data
    df = pd.read_excel('D:\\Private\\my_projects\\python\\final_project_core\\data_source\\e-commerce-dataset.xlsx')
    city_id = df.iloc[0:, 0].values
    raw_X = df.iloc[0:, 2:].values  # dataset
    raw_y = df.iloc[0:, 1].values  # label

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
    # print(ranked_index)
    # print()
    # raw_ranked_index = f_score.f_score(X, y, mode="raw")
    # for i in ranked_index:
    #     print(i, raw_ranked_index[i])

    # X_feature = X.astype(int)
    # y_label = y.astype(int)
    # ranked_index = chi_square.chi_square(X_feature, y_label, mode="index")

    # ranked_index = CFS.cfs(X, y)

    for row in X:
        row_array = []
        for num, feature_idx in enumerate(ranked_index):
            row_array.append(row[feature_idx])
        best_sort_feature.append(row_array)

    best_features = np.array(best_sort_feature)

    si_X = best_features[:, :96]
    X_train, X_test, y_train, y_test = train_test_split(si_X, y, test_size=0.25)

    # Set the parameters by cross-validation
    tuned_parameters = [
        {
            'kernel': ['rbf'],
            'gamma': [1e-3, 1e-4],
            'C': [1, 10, 100, 1000],
            'epsilon': [0.1, 0.5, 1.0, 1.5, 2.0]
        },
        {
            'kernel': ['linear'],
            'C': [1, 10, 100, 1000],
            'epsilon': [0.1, 0.5, 1.0, 1.5, 2.0]
        }
    ]


    clf = GridSearchCV(
        SVR(), tuned_parameters, cv = 3
    )
    clf.fit(X_train, y_train)
    y_true, y_pred = y_test, clf.predict(X_test)

    # Results from Grid Search
    print("\n========================================================")
    print(" Results from Grid Search ")
    print("========================================================")

    print("\n The best estimator across ALL searched params:\n",
          clf.best_estimator_)

    print("\n The best score across ALL searched params:\n",
          clf.best_score_)

    print("\n The best parameters across ALL searched params:\n",
          clf.best_params_)
    print("\nR2 Value:\n")
    print(r2_score(y_true, y_pred))


if __name__ == "__main__":
    main()
