from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import NotFittedError
from sklearn.metrics import confusion_matrix
from sklearn.utils.validation import check_is_fitted
import pandas as pd

from model.temporary.testing_k_folds import get_model_pipeline, TrainTestSplitWrapper
import model.temporary.testing_k_folds as tk
import model.model_setup as ms

class ModelWrapper:
    """
    A model wrapper used to automate running multiple models


    Parameters
    ---------

    model:
    """

    def __init__(self, model,
                 parameters: dict,
                 train_test_wrapper,
                 cv=5):
        self.model_name = model.__name__

        # model will soon become a grid search object, so get the model_name while you can
        self.model = model
        self.parameters = parameters
        self.train_test_wrapper = train_test_wrapper
        self.parameters = parameters

        self.results = dict(matrix=None, mean_cv_score=None, accuracy=None,
                            total_incorrect=None)

        pass

    def use_grid(self, X_train, y_train, random_search=False):
        return tk.grid_search_scores(self.model,
                                  self.parameters,
                                  X_train,
                                  y_train,
                                  random_search=random_search)

    def fit(self, train_test_wrapper, grid_search, max_iter=500, random_search=False,
            scoring='neg_mean_absolute_error',
            repeated_KFold=None):

            with self.train_test_wrapper as splits:
                X_train, X_test, y_train, y_test = splits
                # fit to X_train so X_test has the correct number of columns

                print(type(X_train))

                X_train = pd.get_dummies(X_train)
                X_test = pd.get_dummies(X_test)

                model_pipeline = self.use_grid(X_train, y_train)
                model_pipeline.fit(X_train, y_train)

                predictions = model_pipeline.predict(X_test)

                matrix = confusion_matrix(y_test, predictions)
                self.results["matrix"] = matrix
                self.results["total_incorrect"] = matrix[0][1] + matrix[1][0]

                self.results["accuracy"] = ((y_test.shape[0] - self.results["total_incorrect"]) / y_test.shape[0])

                return model_pipeline, X_test, y_test

    def get_roc_curve(selfs):
        pass

    def get_fit_model(self):
        if self.fitted_model is None:
            raise ValueError("%s has not been fitted yet." % self.model_name)
        else:
            return self.fitted_model

    def get_feature_importance(self):
        pass

    def __str__(self):
        pass

    def is_fit(self):
        try:
            check_is_fitted(self.model)
        except NotFittedError as e:
            print(repr(e))

            return False

        return True


student_data = ms.get_student_data('../data/data.csv', bin=False)
features = ["A8", "A7", "A6", "Gender", "Has_504", "Student on Free or Reduced Lunch", "IEP/Specialized"]

train_wrapper_args = (student_data[features],
                      student_data['ChronicallyAbsent_in_HS'],
                      0.2, # test_size
                      1) # random_state

random_forest = ModelWrapper(RandomForestClassifier(random_state=1),
                             dict(),
                             tk.TrainTestSplitWrapper(*train_wrapper_args))

models_to_run = [random_forest]