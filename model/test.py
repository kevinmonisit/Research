import unittest

from sklearn.tree import DecisionTreeRegressor
from sklearn.pipeline import Pipeline
import model.model_setup as ms
import pandas as pd
import numpy as np


class MyTestCase(unittest.TestCase):

    def setUp(self):
        self.student_data = pd.read_csv("../data/High School East Student Data - Sheet1.csv")
        self.features = ["A6", "A7", "A8"]

        # create column
        self.student_data["AbsencesSum_HS"] = 0

        ms.remove_outliers(self.student_data,
                           self.student_data["AbsencesSum_HS"],
                           lower_bound=0,
                           upper_bound=0.95)

    def basic_tree_model_test(self):

        model_pipeline = Pipeline(steps=[('number_fix', ms.CustomTransformer()),
                                         ('model', DecisionTreeRegressor(random_state=1))
                                         ])

        test = ms.run_test(self.student_data, "AbsencesSum_HS", model_pipeline)

        # these scores come from a prior test run of the decision trees with the same data
        prior_run = pd.np.array([23.43076923, 16.23076923, 15.76923077, 17.93846154, 19.58333333])
        self.assertTrue(np.allclose(test, prior_run, atol=0.001))


if __name__ == '__main__':
    unittest.main()
