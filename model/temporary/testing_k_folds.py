""""
To summarize, there is a bias-variance trade-off associated with the choice of k in k-fold cross-validation.
 Typically, given these considerations, one performs k-fold cross-validation using k = 5 or k = 10,
  as these values have been shown empirically to yield test error rate estimates that suffer neither
  from excessively high bias nor from very high variance.

— Page 184, An Introduction to Statistical Learning, 2013.
https://machinelearningmastery.com/k-fold-cross-validation/
"""

import numpy as np
import pandas as pd
import copy as cp
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, Ridge, ElasticNet
from sklearn.preprocessing import OneHotEncoder, QuantileTransformer
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor

import model.model_setup as ms
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, train_test_split, RepeatedKFold
from sklearn.impute import SimpleImputer
from sklearn.kernel_approximation import RBFSampler


# student_data, pre_process = ms.create_student_data("../../data/High School East Student Data - Sheet1.csv")
def grid_search_scores(estimator, parameter_grid, X, y, cv=3, max_iter=500,
                       random_search=False, scoring='neg_mean_absolute_error',
                       repeated_KFold=None):
    cv_ = cv if repeated_KFold is None else repeated_KFold

    # print(grid.estimator.get_params().keys())
    # transformer = ColumnTransformer(remainder='passthrough',
    #                                 transformers=[('imputer', SimpleImputer(strategy='median',
    #                                                                         fill_value=0,
    #                                                                         missing_values=np.nan),
    #                                                ["A6", "A7", "A8"])])

    pre_process = ColumnTransformer(remainder='passthrough',
                                    transformers=[('categories',
                                                   OneHotEncoder(handle_unknown="error"),
                                                   ["Has_504",
                                                    "Student on Free or Reduced Lunch",
                                                    "IEP/Specialized"])])
    # pre_process.fit(X)
    """

    CustomTransformer fixes the numbers and imputes TRANSFER values.
    It is not in ColumnTransformer in case I want to do Scaling.
    """
    model_pipeline = Pipeline(steps=[#('number_transformer', ms.BinningTransformer()),
                                  #   ('preprocess', pre_process),
#                                     ('RBF', RBFSampler(gamma=0.2)),
                                     ('model', estimator)
                                     ])

    grid = None
    if random_search:
        grid = RandomizedSearchCV(
            random_state=1,
            n_iter=max_iter,
            estimator=model_pipeline,
            param_distributions=parameter_grid,
            #     scoring=scoring,
            cv=cv_,
            n_jobs=-1,
            refit=True
        )
    elif not random_search:
        grid = GridSearchCV(
            estimator=model_pipeline,
            param_grid=parameter_grid,
            #     scoring=scoring,
            cv=cv_,
            n_jobs=-1,
            refit=True
        )

    grid.fit(X, y)

    # cv_results_df = pd.DataFrame(grid.cv_results_)
    #
    # best_index = list(cv_results_df.index[cv_results_df["rank_test_score"] == 1])[0]
    # n_tests = cv if repeated_KFold is None else cv_.get_n_splits()
    # scores = list(cv_results_df.loc[best_index, "split0_test_score":"split%i_test_score" % (n_tests - 1)])
    #
    # print("Estimator: ", grid.estimator['model'])
    # print("Number of iterations: ", max_iter)
    print("Scoring used: ", grid.scoring)
    print("Best parameters: ", grid.best_params_)
    print("Best/Mean score using best parameters: ", grid.best_score_)
    # print("Variance: ", np.var(scores))

    return grid


class TrainTestSplitWrapper:

    def __init__(self, X, y, random_state=1, test_size=0.2):
        self.X = X
        self.y = y
        self.random_state = random_state
        self.test_size = test_size

    def __enter__(self):
        X_ = cp.deepcopy(self.X)
        y_ = cp.deepcopy(self.y)
        return train_test_split(X_, y_, test_size=self.test_size)

    def __exit__(self, *args, **kwargs):
        pass


absence_rates = ms.get_column_rates("A", 6, 9)
features = ["Has_504", "A8", "Has_504", "Student on Free or Reduced Lunch", "IEP/Specialized", "ChronicallyAbsent_in_MS"]

# for rate in absence_rates:
#     features.append(rate)

# features = ["A6", "A7", "A8", "T6", "T7", "T8"]

student_data = ms.get_student_data("../../data/data.csv", bin=True)

data_split = TrainTestSplitWrapper(student_data[features],
                                   student_data['ChronicallyAbsent_in_HS'],
                                   test_size=0.2, random_state=1)

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import average_precision_score
from sklearn import svm
from sklearn.linear_model import LogisticRegression

def column_list(letter, start, end):
    return ["%s%d" % (letter, i) for i in range(start, end)]



with data_split as splits:
    X_train, X_test, y_train, y_test = splits
    # fit to X_train so X_test has the correct number of columns

    print(type(X_train))

    pre_process = ColumnTransformer(remainder='passthrough',
                                    transformers=[('categories',
                                                   OneHotEncoder(handle_unknown="error"),
                                                   ["Has_504",
                                                    "Student on Free or Reduced Lunch",
                                                    "IEP/Specialized"])])

    X_train = pd.get_dummies(X_train)
    X_test = pd.get_dummies(X_test)

    #
    # X_train = pre_process.transform(X_train)
    # X_test = pre_process.transform(X_test)

    # \Best
    # pipeline: DecisionTreeClassifier(RBFSampler(input_matrix, gamma=0.2), criterion=entropy, max_depth=5,
    #                                  min_samples_leaf=12, min_samples_split=10)
    # 512.8246681690216
    # [[7 1]Prepare the training data
    #  [1 7]]
    # 16
    #
    # Best
    # parameters: {'model__criterion': 'entropy', 'model__max_depth': 3, 'model__min_samples_leaf': 1,
    #              'model__min_samples_split': 8, 'number_transformer__bins': 5}
    # Best / Mean
    # score
    # using
    # best
    # parameters: 0.8396825396825397

    decision_tree_grid = dict(#number_transformer__bins=range(1, 8),
                              model__min_samples_leaf=range(2,13),
                              model__min_samples_split=range(3,11),
                              model__max_depth=range(2,7))

    # d_tree = Pipeline(steps=[#('RBF', RBFSampler(gamma=0.2)),
    #                          ('model', DecisionTreeClassifier(max_depth=3,
    #                                                           min_samples_leaf=12,
    #                                                           min_samples_split=10,
    #                                                           criterion='entropy'))
    #                          ])

    # d_tree.fit(X_train, y_train)
    d_tree = grid_search_scores(DecisionTreeClassifier(random_state=1, criterion='entropy'),
                                decision_tree_grid,
                                X_train,
                                y_train,
                                random_search=False)

     # repeated_KFold=RepeatedKFold(random_state=1,
     #                             n_splits=5,
     #                              n_repeats=3))

    print("PREDICTIONS==========================")
    predictions = d_tree.predict(X_test)
    print(predictions)
    print(y_test)
    matrix = confusion_matrix(y_test, predictions)
    print(matrix)
    total_incorrect = matrix[0][1] + matrix[1][0]
    print("Percentage: ", (y_test.shape[0] - total_incorrect) / y_test.shape[0])
    print(len(y_test))
    print("=======")
    # importance = d_tree['model'].feature_importances_
    importance = d_tree.best_estimator_['model'].feature_importances_

    # summarize feature importance
    for i, v in enumerate(importance):
        print('Feature: %0d, Score: %.5f' % (i, v))
