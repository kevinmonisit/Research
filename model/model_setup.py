import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer

pd.options.mode.chained_assignment = None  # default='warn'


class CustomTransformer(BaseEstimator, TransformerMixin):

    # set new_value to None if Pipeline contains SimpleImputer
    # this is for absences and tardies since some students are
    # transfer students. The placeholder in the CSV is the string "TRANSFER"
    def __init__(self, new_value=0, bins=1, transformation="fixed", using_imputer=False):
        self.new_value = new_value
        self.transformation = transformation
        self.bins = bins
        self.using_imputer = using_imputer

        if self.bins == 0:
            raise ValueError("Bins cannot be set to 0")

        if self.new_value is None and self.using_imputer is None:
            raise ValueError("New value has been set to None but imputer strategy is still None.")

    def fit(self, x, y=None):
        return self

    def transform(self, df):

        imputer = None

        if self.using_imputer:
            imputer = SimpleImputer(strategy=self.using_imputer)

        for x in df:
            if self.transformation == "fixed":
                df[x] = df[x].apply(transform_value, args=(imputer, self.bins, self.new_value))

            elif self.transformation == "log":
                df[x] = df[x].apply(lambda j: np.log((1 + convert_stat(j, new_value=self.new_value))))

            else:
                raise Exception("Transformation argument was not correctly assigned.")

        return df


def print_scores(score_array):
    print("Scores: ", score_array)
    print("Score mean: ", np.mean(score_array))
    print("Score variance: ", np.var(score_array))


# easy way of accessing A_6, A_7, ... A_N columns
def column_list(letter, start, end):
    return ["%s%d" % (letter, i) for i in range(start, end)]


def remove_outliers(column, target, lower_bound: float = 0., upper_bound: float = 100.):
    non_outliers = target.between(target.quantile(lower_bound), target.quantile(upper_bound))
    count = 0

    for index in range(0, len(column)):
        if ~non_outliers[index]:
            count += 1
            column.drop(index, inplace=True)

    print("%i outliers were removed" % count)

    return count


# convert strings to int type even if it's a float
def convert_stat(x, new_value=0):
    if not isinstance(x, int):

        if not isinstance(x, float) and '.' not in x:
            return new_value if x == "TRANSFER" else int(x)
        else:
            return new_value if x == "TRANSFER" else int(float(x))

    else:
        return x


# for using imputer and binning. can't use convert_stat only,
# since you can't divide None by anything
def transform_value(x, imputer=False, bins=1, new_value=0):
    value = convert_stat(x, new_value=new_value)

    """
    If imputer is not None, then new_value should be set to some value
    that the imputer will use to identify cells.
    
    The imputer still needs to be set in the pipeline!!
    """
    return np.floor(value / float(bins)) if not imputer else value


def run_test(data, target, pipeline, features_=None):
    if features_ is None:
        features_ = ["A6", "A7", "A8"]

    scores = -1 * cross_val_score(pipeline,
                                  data[features_],
                                  data[target],
                                  cv=5,
                                  scoring='neg_mean_absolute_error')
    return scores


def create_student_data(path, lower_bound: float = 0, upper_bound: float = 0.95):
    student_data = pd.read_csv(path)

    student_data["AbsencesSum_HS"] = 0

    # Pipeline doesn't allow transformations on the target label
    # so I have to do transformations outside of Pipeline in order
    # to sum all absences in High School for each student.
    for j in column_list("A", 9, 13):
        student_data[j] = student_data[j].apply(convert_stat)

    student_data["AbsencesSum_HS"] = student_data[column_list('A', 9, 13)].sum(axis=1)

    # because we've created the total absences in high school column
    # we are now able to eliminate outliers in the dataset.
    remove_outliers(student_data, student_data["AbsencesSum_HS"], lower_bound, upper_bound)

    pre_process = ColumnTransformer(remainder='passthrough',
                                    transformers=[('categories',
                                                   OneHotEncoder(),
                                                   ["Gender", "IEP/Specialized"])])

    return student_data, pre_process
