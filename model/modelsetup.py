
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, KBinsDiscretizer
from sklearn.tree import DecisionTreeRegressor
pd.options.mode.chained_assignment = None  # default='warn'


# easy way of accessing A_6, A_7, ... A_N columns
def column_list(letter, start, end):
    return ["%s%d" % (letter, i) for i in range(start, end)]


def remove_outliers(column, target, first, second):
    non_outliers = target.between(target.quantile(first), target.quantile(second))

    for index in range(0, len(column)):
        if ~non_outliers[index]:
            column.drop(index, inplace=True)


# convert strings to int type even if it's a float
# replace by median or mean?
def convert_stat(x, new_value=0):
    if not isinstance(x, int):

        if not isinstance(x, float) and '.' not in x:
            return new_value if x == "TRANSFER" else int(x)
        else:
            return new_value if x == "TRANSFER" else int(float(x))

    else:
        return x


class SumTransformer(BaseEstimator):

    # set new_value to None if Pipeline contains SimpleImputer
    # this is for absences and tardies since somet students are
    # transfer students. The placeholder in the CSV is the string "TRANSFER"
    def __init__(self, new_value=0, bins=1):
        self.new_value = new_value
        self.bins = bins

    def fit(self, x, y=None):
        return self

    def transform(self, df):
        # change to for i in ["A", "T"] to include tardies if need be
        for i in ["A"]:

            # corrects the values in the data frame that will be used in the training models
            for j in column_list(i, 6, 9):
                df[j] = df[j].apply(convert_stat, self.new_value)
                df[j] = np.array(np.floor(np.array(df[j]) / float(self.bins)))

        return df


def run_test(data, target, pipeline, features_=["A6", "A7", "A8"]):

    scores = -1 * cross_val_score(pipeline,
                              data[features_],
                              data[target],
                              cv=5,
                              scoring='neg_mean_absolute_error')

    return scores


student_data = pd.read_csv("../data/High School East Student Data - Sheet1.csv")
features = ["A6", "A7", "A8", "Gender", "IEP/Specialized"]
student_data["AbsencesSum_HS"] = 0

# Pipeline doesn't allow transformations on the target label
# so I have to do transformations outside of Pipeline in order
# to sum all absences in High School for each student.
for j in column_list("A", 9, 13):
    student_data[j] = student_data[j].apply(convert_stat)

student_data["AbsencesSum_HS"] = student_data[column_list('A', 9, 13)].sum(axis=1)

# because we've created the total absences in high school column
# we are now able to eliminate outliers in the dataset.
remove_outliers(student_data, student_data["AbsencesSum_HS"], 0, 0.95)

pre_process = ColumnTransformer(remainder='passthrough',
                                transformers=[('categories', OneHotEncoder(), ["Gender", "IEP/Specialized"])])

model_pipeline = Pipeline(steps=[('number_fix', SumTransformer(bins=5)),
                                 ('model', DecisionTreeRegressor(random_state=1))
                                 ])

### Sorta like unit testing but in jupyter
# FIX THIS BECAUSE IF FEATURES NEED TO BE CHANGED, IT CANNOT BE FIXED
test = run_test(student_data, "AbsencesSum_HS", model_pipeline)
prior_run = np.array([23.43076923, 16.23076923, 15.76923077, 17.93846154, 19.58333333])

if not np.allclose(test, prior_run, atol=0.001):
    raise Exception("Modification to pre-processing led to unintended results.")


