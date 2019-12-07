import numpy as np
from sklearn.svm import SVR
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import LeaveOneOut

def trainf(X, y):
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

    return best_pred, best_score, result, ten_column_predictions
