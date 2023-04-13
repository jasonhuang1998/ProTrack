import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV, KFold, train_test_split
from sklearn.tree import DecisionTreeClassifier as DTC
from module.model import model
from module.log import printLog

def runModel(config):
    TrainData1 = pd.read_csv(config['model']['input_file_location'], on_bad_lines='skip')
    TrainData1=TrainData1.replace([np.inf, -np.inf], np.nan)

    TrainData=TrainData1.set_index('pair')
    TrainData=TrainData.fillna(0)

    X = TrainData.copy().drop(['State'], axis=1)
    y = TrainData['State']
    Shift_q=X.copy().iloc[:, 0:21]

    Shift_q['max_index']=Shift_q.idxmax(axis=1)
    X['max_index']=Shift_q['max_index']
    X['max_index']=X['max_index'].replace('CosSim_ble', '0')
    X['max_index']=X['max_index'].map(lambda x: x.lstrip('CosSim_ble_shift'))
    X['max_index']=X['max_index'].astype(int)
    X['max_index']=X['max_index'].abs()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
    dtc = fit_model_k_fold(X_train, y_train)
    dtc.fit(X_train, y_train)
    predict_target = dtc.predict(X_test)


    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
    clf = model().clf
    
    clf.fit(X_train,y_train)
    predict_target = clf.predict(X_test)
    print(metrics.classification_report(y_test, predict_target, target_names=['Independent','Companion','Follower']))
    print('XGB Accuracy:', clf.score(X_test, y_test))



def fit_model_k_fold(X, y):
    """ Performs grid search over the 'max_depth' parameter for a
        decision tree regressor trained on the input data [X, y]. """

    # Create cross-validation sets from the training data
    # cv_sets = ShuffleSplit(n_splits = 10, test_size = 0.20, random_state = 0)
    k_fold = KFold(n_splits=10)

    #  Create a decision tree clf object
    dtc = DTC()

    # params = {'max_depth':range(10,21),'criterion':np.array(['entropy','gini'])}
    params = {'max_depth':range(5,50),'criterion':np.array(['entropy','gini'])}

    # Transform 'accuracy_score' into a scoring function using 'make_scorer'
    scoring_fnc = make_scorer(accuracy_score)

    # Create the grid search object
    grid = GridSearchCV(dtc, param_grid=params,scoring=scoring_fnc,cv=k_fold,n_jobs=8)

    # Fit the grid search object to the data to compute the optimal model
    grid = grid.fit(X, y)

    # Return the optimal model after fitting the data
    return grid.best_estimator_



def accuracy_score(truth, pred):
    """ Returns accuracy score for input truth and predictions. """
    # Ensure that the number of predictions matches number of outcomes
    if len(truth) == len(pred):
        # Calculate and return the accuracy as a percent
        return(truth == pred).mean()*100

    else:
        return 0