
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn import svm
from imblearn.combine import SMOTETomek
from imblearn.under_sampling import ClusterCentroids
from imblearn.over_sampling import SMOTE
import xgboost
import numpy as np


def course_hyper_parameter_cv(sampling_method, model_name, scores, k, x_train, y_train, x_test, y_test):
    """
    Performs a course grid search with 5-fold cross-validation for a given set of models and scoring methods
    :param sampling_method: Options for re-sampling the training data if the target is imbalanced
    (currently supports 'smote', 'smote_0.5' (ratio=0.5), 'cluster_centroids', 
    cluster_centroids_0.5' (ratio=0.5), and 'smote_tomek' from imblearn package)
    :param model_name: Model name as a string (currently supports 'RandomForestClassifier', 'GradientBoostingClassifier',
    'AdaBoostClassifier', and 'svm' all from sklearn)
    :param scores: Options for scoring metrics e.g. f1_weighted, roc_auc, accuracy_score (sklearn metrics)
    :param k: Number of folds for cross-validation (e.g. 5)
    :param x_train: training data for features
    :param y_train: training data for target
    :param x_test: test data for features
    :param y_test: test data for target
    :return: Prints the best parameters for each scoring metric, sampling method, and model
    """

    # Select model

    if model_name == 'RandomForestClassifier':
        model = RandomForestClassifier(n_jobs=-1, oob_score=True, criterion='entropy')

        tuning_params = [{'n_estimators': [500, 1000, 2000],
                          'max_features': ['sqrt', 'auto', None],
                          'max_depth': [5, 10, 20]
                          # 'min_samples_split': [2, 5, 10]
                          }]

    elif model_name == 'GradientBoostingClassifier':
        model = GradientBoostingClassifier()

        tuning_params = [{'n_estimators': [500, 1000, 2000],
                          'max_features': ['sqrt', 'auto', None],
                          'max_depth': [1, 2, 5, 10],
                          # 'min_samples_split': [2, 5, 10],
                          'learning rate': [0.01, 0.1]}]

    elif model_name == 'xgboost':
        model = xgboost.XGBClassifier()
        scale_target = sum(np.argwhere(y_train == 0))/sum(np.argwhere(y_train == 1))

        tuning_params = [{'n_estimators': [500, 1000, 2000],
                          'max_depth': [3, 5, 10],
                          'colsample_bytree': [0.5, 1],
                          'max_delta_step': [0, 5, 10],
                          'gamma': [0, 5],
                          'min_child_weight': [1, 5],
                          'scale_pos_weight': [1, scale_target[0]],
                          'learning rate': [0.01, 0.1]}]

    elif model_name == 'AdaBoostClassifier':
        model = AdaBoostClassifier()

        tuning_params = [{'n_estimators': [500, 1000, 2000],
                          'learning rate': [0.001, 0.01, 0.1]}]

    elif model_name == 'svm':
        model = svm.SVC()

        tuning_params = [{'kernel': ['linear', 'rbf'],
                          'C': [0.01, 0.1, 1, 10],
                          'gamma': [0.01, 0.1, 1, 10]}]

    else:
        print('Unknown model')

    # Select re-sampling method

    if sampling_method == 'none':
        x_r, y_r = x_train, y_train

    elif sampling_method == 'smote':
        sm = SMOTE(random_state=26)
        x_r, y_r = sm.fit_sample(x_train, y_train)

    elif sampling_method == 'smote_0.5':
        sm = SMOTE(random_state=26, ratio=0.5)
        x_r, y_r = sm.fit_sample(x_train, y_train)

    elif sampling_method == 'cluster_centroids':
        cc = ClusterCentroids(random_state=26)
        x_r, y_r = cc.fit_sample(x_train, y_train)

    elif sampling_method == 'cluster_centroids_0.5':
        cc = ClusterCentroids(random_state=26, ratio=0.5)
        x_r, y_r = cc.fit_sample(x_train, y_train)

    elif sampling_method == 'smote_tomek':
        st = SMOTETomek(random_state=26)
        x_r, y_r = st.fit_sample(x_train, y_train)

    # Cross validation

    for score in scores:
        print('\nHyper-parameter tuning for {}\n'.format(score))

        clf = GridSearchCV(model, tuning_params, refit=True, cv=k, scoring=score)
        clf.fit(x_r, y_r)
        print('Best Parameters:')
        print(clf.best_params_)
        print('\nDetailed report:\n')
        y_true, y_predicted = y_test, clf.predict(x_test)
        print(classification_report(y_true, y_predicted))

    return clf
