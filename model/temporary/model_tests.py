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


# student_data, pre_process = ms.create_student_data("../../data/High School East Student Data - Sheet1.csv")
def grid_search_scores(estimator, parameter_grid, X, y, cv=5, max_iter=1,
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


    model_pipeline = Pipeline(steps=[('number_transformer', ms.BinningTransformer()),
                               #      ('preprocess', pre_process),
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
            scoring=scoring,
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
    # print("Scoring used: ", grid.scoring)
    # print("Best parameters: ", grid.best_params_)
    # print("Best/Mean score using best parameters: ", grid.best_score_)
    # print("Variance: ", np.var(scores))

    return grid


"""
TODO:
the with statement can be repeated. clean it up for a bit
"""


class TrainTestSplitWrapper:

    def __init__(self, X, y, random_state=1, test_size=0.2):
        self.X = X
        self.y = y
        self.random_state = random_state
        self.test_size = test_size

    def __enter__(self):
        X_ = cp.deepcopy(self.X)
        y_ = cp.deepcopy(self.y)
        return train_test_split(X_, y_, random_state=self.random_state, test_size=self.test_size)

    def __exit__(self, *args, **kwargs):
        pass


#features = ["A6", "A7", "A8", "T6", "T7", "T8", "Has_504", "Student on Free or Reduced Lunch",
  #          "IEP/Specialized"]

features = ["A6", "A7", "A8", "T6", "T7", "T8"]

student_data = ms.get_student_data("../../data/data.csv")

data_split = TrainTestSplitWrapper(student_data[features],
                                   student_data['ChronicallyAbsent_in_HS'],
                                   random_state=1, test_size=0.2)

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import average_precision_score
from sklearn import svm
from sklearn.linear_model import LogisticRegression

with data_split as splits:
    X_train, X_test, y_train, y_test = splits
    # fit to X_train so X_test has the correct number of columns

    print(type(X_train))
    X_train = pd.get_dummies(X_train)

    # pre_process.fit(X_train)

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
    decision_tree_grid = dict(number_transformer__bins=range(1, 12))
                       #       model__min_samples_leaf=[12],
                      #        model__min_samples_split=[10],
                     #         model__max_depth=[5])

    d_tree = grid_search_scores(svm.SVC(),
                                decision_tree_grid,
                                X_train,
                                y_train)

    print("PREDICTIONS==========================")
    predictions = d_tree.predict(X_test)
    print(predictions)
    print(y_test)

    print(confusion_matrix(y_test, predictions))
    print(len(y_test))

"""
When grid search is performed, X_train, X_test, ..., are modified, so they must be separated.
"""

# # LINEAR GRID
# with data_split as splits:
#
#     X_train, X_test, y_train, y_test = splits
#
#     linear_grid = dict(model__normalize=[True, False],
#                        number_transformer__bins=range(1, 20),
#                        transform__imputer__strategy=['mean', 'most_frequent', 'median', 'constant'])
#
#     linear = grid_search_scores(LinearRegression(), linear_grid, X_train, y_train,
#                                 scoring='neg_mean_absolute_error', random_search=False)
#
#     print("Coefficients: ", linear.best_estimator_['model'].coef_)
#     print("Score: ", linear.score(X_test, y_test))


# SVR TEST
# with data_split as splits:
#     X_train, X_test, y_train, y_test = splits
#
#     svr_grid = dict(number_transformer__bins=range(1, 12),
#                     transform__imputer__strategy=['mean', 'most_frequent', 'median', 'constant'],
#                     model__kernel=['linear', 'poly', 'rbf', 'sigmoid'],
#                     model__C=[0.001, 0.01, 1, 10, 100],
#                     model__degree=[1, 2, 6],
#                     model__coef0=[0.01, 10, 0.5],
#                     model__gamma=['auto', 'scale'])

#  svr = grid_search_scores(SVR(), svr_grid,
#                           X_train, y_train, random_search=True,
#                           max_iter=10,
#                          repeated_KFold=RepeatedKFold(random_state=1,
#                                                        n_splits=5,
#                                                        n_repeats=3))

# predictions = svr.predict(X_test)
# print("PREDICTIONS of SVR::::::")
# print(list(zip(predictions, y_test)))
# print(svr.score(X_test, y_test))


# RANDOM FOREST TEST
# with data_split as splits:
#     X_train, X_test, y_train, y_test = splits
#
#     # https://www.analyticsvidhya.com/blog/2020/03/beginners-guide-random-forest-hyperparameter-tuning/
#     d_tree_grid = dict(model__max_depth=range(1, 50, 5),
#                        model__n_estimators=range(1, 100, 5),
#                        model__max_features=['auto', 'sqrt', 'log2', None],
#                        number_transformer__bins=range(1, 15),
#                        transform__imputer__strategy=['mean', 'most_frequent', 'median', 'constant'])
#   #  print("RANDOM FOREST TEST \n========\n========\n========")
# d_tree = grid_search_scores(RandomForestRegressor(random_state=1), d_tree_grid,
#                             X_train, y_train, random_search=True,
#                              max_iter=1,
#                             repeated_KFold=RepeatedKFold(random_state=1,
#                                                          n_splits=5,
#                                                           n_repeats=5))

# print(d_tree.score(X_test, y_test))


# RANDOM FOREST
# with data_split as splits:
#     X_train, X_test, y_train, y_test = splits
#
#     # https://www.analyticsvidhya.com/blog/2020/03/beginners-guide-random-forest-hyperparameter-tuning/
#     d_tree_grid = dict(model__max_depth=range(1, 50, 5),
#                        model__n_estimators=range(1, 100, 5),
#                        model__max_features=['auto', 'sqrt', 'log2', None],
#                        number_transformer__bins=range(1, 15),
#                        transform__imputer__strategy=['mean', 'most_frequent', 'median', 'constant'])

# d_tree = grid_search_scores(RandomForestRegressor(random_state=1), d_tree_grid,
#                             X_train, y_train, random_search=True)

# d_tree.score(X_test, y_test)

# ELASTIC NET
# with data_split as splits:
#
#     X_train, X_test, y_train, y_test = splits
#
#     elastic_net_grid = dict(model__normalize=[True],
#                             model__alpha=np.logspace(0.001, 2, 20),
#                             model__l1_ratio=np.arange(0, 1.1, 0.1),
#                             model__max_iter=[5000],
#                             number_transformer__bins=range(1, 20),
#                             transform__imputer__strategy=['mean', 'most_frequent', 'median', 'constant'])

#  elastic_net = grid_search_scores(ElasticNet(random_state=1), elastic_net_grid, X_train, y_train,
#                                  scoring='neg_mean_absolute_error',
#                                   random_search=False,
#                                 max_iter=1)

# print("Coefficients: ", linear.best_estimator_['model'].coef_)
# print("Score: ", elastic_net.score(X_test, y_test))


"""
ridge_grid = dict(model__alpha=list(np.logspace(0.01, 2, 25)),
                  preprocessor__transform__imputer_strategy=["mean", "median"],
                  preprocessor__transform__bins=range(1, 20))

#ridge = grid_search_scores(Ridge(random_state=1), ridge_grid)
#-17.7802986264604, -14.751912705849396

elastic_grid = dict(model__l1_ratio=np.arange(0, 1, 25),
                    model__alpha=np.logspace(0.001, 10, 25))
#elastic_net = grid_search_scores(ElasticNet(random_state=1, max_iter=10000, normalize=True), elastic_grid)


https://kazemnejad.com/blog/how_to_do_deep_learning_research_with_absolutely_no_gpus_part_2/
"""
