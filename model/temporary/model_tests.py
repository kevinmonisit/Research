# i have this so i don't have to keep running a jupyter server

import numpy as np
import pandas as pd
import copy as cp
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, Ridge, ElasticNet
from sklearn.preprocessing import OneHotEncoder, QuantileTransformer
from sklearn.tree import DecisionTreeRegressor

import model.model_setup as ms
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, train_test_split
from sklearn.impute import SimpleImputer

student_data, pre_process = ms.create_student_data("../../data/High School East Student Data - Sheet1.csv")
features = ["A6", "A7", "A8"]


def grid_search_scores(estimator, parameter_grid, X, y, cv=5, max_iter=20,
                       random_search=False, scoring='neg_mean_absolute_error'):

    # print(grid.estimator.get_params().keys())
    transformer = ColumnTransformer(remainder='passthrough',
                                    transformers=[('imputer', SimpleImputer(strategy='mean',
                                                                            fill_value=0,
                                                                            missing_values=np.nan),
                                                   ["A6", "A7", "A8"])])

    """
    
    CustomTransformer fixes the numbers and imputes TRANSFER values.
    It is not in ColumnTransformer in case I want to do Scaling.
    """
    model_pipeline = Pipeline(steps=[('number_transformer', ms.CustomTransformer(new_value=np.nan,
                                                                                 using_imputer=True)),
                                     ('transform', transformer),
                                     ('model', estimator)
                                     ])

    grid = None

    if random_search:
        grid = RandomizedSearchCV(
            random_state=1,
            n_iter=max_iter,
            estimator=model_pipeline,
            param_distributions=parameter_grid,
            scoring=scoring,
            cv=cv,
            n_jobs=-1,
            refit=True
        )
    elif not random_search:
        grid = GridSearchCV(
            estimator=model_pipeline,
            param_grid=parameter_grid,
            scoring=scoring,
            cv=cv,
            n_jobs=-1,
            refit=True
        )

    grid.fit(X, y)

    cv_results_df = pd.DataFrame(grid.cv_results_)

    best_index = list(cv_results_df.index[cv_results_df["rank_test_score"] == 1])[0]
    scores = list(cv_results_df.loc[best_index, "split0_test_score":"split%i_test_score" % (cv-1)])

    print("Estimator: ", grid.estimator['model'])
    print("Number of iterations: ", max_iter)
    print("Scoring used: ", grid.scoring)
    print("Best parameters: ", grid.best_params_)
    print("Best/Mean score using best parameters: ", grid.best_score_)
    print("Variance: ", np.var(scores))

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


data_split = TrainTestSplitWrapper(student_data[features],
                                   student_data["AbsencesSum_HS"],
                                   random_state=1, test_size=0.2)

"""
When grid search is performed, X_train, X_test, ..., are modified, so they must be separated.
"""

# LINEAR GRID
with data_split as splits:

  #  X_train, X_test, y_train, y_test = splits

    linear_grid = dict(model__normalize=[True, False],
                       number_transformer__bins=range(1, 20),
                       transform__imputer__strategy=['mean', 'most_frequent', 'median', 'constant'])

   # linear = grid_search_scores(LinearRegression(), linear_grid, X_train, y_train,
 #                               scoring='neg_mean_absolute_error')

  #  print("Coefficients: ", linear.best_estimator_['model'].coef_)
  #  print("Score: ", linear.score(X_test, y_test))


# RANDOM FOREST
with data_split as splits:
   # X_train, X_test, y_train, y_test = splits

    # https://www.analyticsvidhya.com/blog/2020/03/beginners-guide-random-forest-hyperparameter-tuning/
    d_tree_grid = dict(model__max_depth=range(1, 50, 5),
                       model__n_estimators=range(1, 100, 5),
                       model__max_features=['auto', 'sqrt', 'log2', None],
                       number_transformer__bins=range(1, 20, 5),
                       transform__imputer__strategy=['mean', 'most_frequent', 'median', 'constant'])

   # d_tree = grid_search_scores(RandomForestRegressor(random_state=1), d_tree_grid,
    #                            X_train, y_train, random_search=True)

# ELASTIC NET
with data_split as splits:

    X_train, X_test, y_train, y_test = splits

    elastic_net_grid = dict(model__normalize=[True],
                            model__alpha=np.logspace(0.001, 2, 20),
                            model__l1_ratio=np.arange(0, 1.1, 0.1),
                            model__max_iter=[5000],
                            number_transformer__bins=range(1, 20),
                            transform__imputer__strategy=['mean', 'most_frequent', 'median', 'constant'])

    elastic_net = grid_search_scores(ElasticNet(random_state=1), elastic_net_grid, X_train, y_train,
                                     scoring='neg_mean_absolute_error',
                                     random_search=False,
                                     max_iter=500)

   # print("Coefficients: ", linear.best_estimator_['model'].coef_)
    print("Score: ", elastic_net.score(X_test, y_test))


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