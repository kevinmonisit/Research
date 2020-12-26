from sklearn.exceptions import NotFittedError
from sklearn.utils.validation import check_is_fitted


class ModelWrapper:
    """
    A model wrapper used to automate running multiple models


    Parameters
    ---------

    model:
    """

    results = dict(matrix=None, mean_cv_score=None, accuracy=None,
                   total_incorrect=None)

    parameters = None
    model = None
    model_name = None

    def __init__(self, model, parameters: dict, ):
        self.model_name = model.__name__
        self.model = model

        self.parameters = parameters

        pass

    def fit(self):
        pass

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
