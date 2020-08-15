# i have this so i don't have to keep running a jupyter server

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, Ridge, ElasticNet
from sklearn.preprocessing import OneHotEncoder, QuantileTransformer
import model.model_setup as ms
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import GridSearchCV
from sklearn.impute import SimpleImputer

student_data, pre_process = ms.create_student_data("../../data/High School East Student Data - Sheet1.csv")
features = ["A6", "A7", "A8"]

print(student_data.shape)


def grid_search_scores(estimator, parameter_grid):

    # print(grid.estimator.get_params().keys())
    transformer = ColumnTransformer(remainder='passthrough',
                                    transformers=[('transform', ms.CustomTransformer(new_value=0),
                                                   ["A6", "A7", "A8"])])

                                             #     ('imputer', SimpleImputer(),
                                              #     ["A6", "A7", "A8"])])

    model_pipeline = Pipeline(steps=[('preprocessor', transformer),
                                     ('model', estimator)
                                     ])

    grid = GridSearchCV(
        estimator=model_pipeline,
        param_grid=parameter_grid,
        scoring='neg_mean_absolute_error',
        cv=5,
        n_jobs=-1,
        refit=True
    )

    grid.fit(student_data[features], student_data["AbsencesSum_HS"])

    cv_results_df = pd.DataFrame(grid.cv_results_)

    best_index = list(cv_results_df.index[cv_results_df["rank_test_score"] == 1])[0]
    scores = cv_results_df.loc[best_index, "split0_test_score":"split4_test_score"]

    print("Estimator: ", grid.estimator['model'])
    print("Best parameters: ", grid.best_params_)
    print("Best score: ", grid.best_score_)
    print("Variance: ", np.var(list(scores)))

    return grid


linear_grid = dict(model__normalize=[True, False],
                   preprocessor__transform__bins=range(2, 20))
                   #preprocessor__transform__imputer_strategy=[None, "mean", "median"])

linear = grid_search_scores(LinearRegression(), linear_grid)

print("Coefficients: ", linear.best_estimator_['model'].coef_)


"""
Estimator:  LinearRegression()
Best parameters:  {'model__normalize': True, 'preprocessor__transform__bins': 8}
Best score:  -13.581855686902696
Variance:  0.7745412719153173
Coefficients:  [-0.65948776  2.59962559 19.99600742]
"""

"""
ridge_grid = dict(model__alpha=list(np.logspace(0.01, 2, 25)),
                  preprocessor__transform__imputer_strategy=["mean", "median"],
                  preprocessor__transform__bins=range(1, 20))

#ridge = grid_search_scores(Ridge(random_state=1), ridge_grid)
#-17.7802986264604, -14.751912705849396

elastic_grid = dict(model__l1_ratio=np.arange(0, 1, 25),
                    model__alpha=np.logspace(0.001, 10, 25))
#elastic_net = grid_search_scores(ElasticNet(random_state=1, max_iter=10000, normalize=True), elastic_grid)

"""