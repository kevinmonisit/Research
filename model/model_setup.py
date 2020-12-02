import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import OneHotEncoder

pd.options.mode.chained_assignment = None  # default='warn'


class BinningTransformer(BaseEstimator, TransformerMixin):

    # set new_value to None if Pipeline contains SimpleImputer
    # this is for absences and tardies since some students are
    # transfer students. The placeholder in the CSV is the string "TRANSFER"
    def __init__(self, bins=1):
        self.bins = bins

        if self.bins == 0:
            raise ValueError("Bins cannot be set to 0")

    def fit(self, x, y=None):
        return self

    """
    Transformers all the values and converts strings and "TRANSFER" to something legitimate.

    """

    def transform(self, df):

        # for i in ["A", "T"]:
        #     for j in column_list(i, 6, 9):
        #         df[j] = df[j].apply(lambda x: int(x / self.bins))

        for j in ["A6", "A7", "A8", "T8"]:
            df[j] = df[j].apply(lambda x: int(x / self.bins))

        return df


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

    #  if np.isnan(self.new_value) and not self.using_imputer:
    #     raise Exception("If new_value is nan, then an imputer must be used. If being used, set",
    #                    "using_imputer to True")

    def fit(self, x, y=None):
        return self

    """
    Transformers all the values and converts strings and "TRANSFER" to something legitimate.

    """

    def transform(self, df):

        for i in column_list("A", 6, 9):
            if self.transformation == "fixed":
                df[i] = df[i].apply(convert_absence_columns, args=(self.new_value, self.bins))

            elif self.transformation == "log":
                df[i] = df[i].apply(lambda j: np.log((1 + convert_absence_columns(j, self.new_value, self.bins))))

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


def get_column_rates(letter, start, end):
    return ["%s%d_rate" % (letter, i) for i in range(start, end)]


def remove_outliers(column, target, lower_bound: float = 0., upper_bound: float = 100.):
    non_outliers = target.between(target.quantile(lower_bound), target.quantile(upper_bound))
    count = 0

    for index in range(0, len(column)):
        if ~non_outliers[index]:
            count += 1
            column.drop(index, inplace=True)

    print("%i outliers were removed" % count)

    return count


def convert_absence_columns(x, new_value=0, bins=1):
    """
    Columns that contain "TRANSFER" are converted to strings, so they must be converted.
    Also, "TRANSFER" cells are considered to be missing values, so they are assigned
    to a new_value.
    """
    if x != "TRANSFER":
        try:
            return np.floor(int(float(x)) / float(bins))
        except ValueError:
            print("Absence cell is neither \"TRANSFER\" or a number.")

    elif x == "TRANSFER":
        return new_value


def run_test(data, target, pipeline, features_=None):
    if features_ is None:
        features_ = ["A6", "A7", "A8"]

    scores = -1 * cross_val_score(pipeline,
                                  data[features_],
                                  data[target],
                                  cv=5,
                                  scoring='neg_mean_absolute_error')
    return scores


"""
There are some critical preprocessing that goes on before the test split.
Although it happens before the test split, they are general enough that
it will not cause data leakage.

Creating a "AbsencesSum_HS" column is impossible to do when working with the pipeline.
Thus, it is beyond my control. Therefore, I must transform and impute. The binning is
done in the Pipeline, though.
"""


def create_student_data(path, lower_bound: float = 0, upper_bound: float = 0.95):
    student_data = pd.read_csv(path)

    student_data["AbsencesSum_HS"] = 0

    # Pipeline doesn't allow transformations on the target label
    # so I have to do transformations outside of Pipeline in order
    # to sum all absences in High School for each student.
    for j in column_list("A", 9, 13):
        student_data[j] = student_data[j].apply(convert_absence_columns)

    student_data["AbsencesSum_HS"] = student_data[column_list('A', 9, 13)].sum(axis=1)

    # because we've created the total absences in high school column
    # we are now able to eliminate outliers in the dataset.
    remove_outliers(student_data, student_data["AbsencesSum_HS"], lower_bound, upper_bound)

    pre_process = ColumnTransformer(remainder='passthrough',
                                    transformers=[('categories',
                                                   OneHotEncoder(),
                                                   ["Gender", "IEP/Specialized"])])

    return student_data, pre_process


# convert strings to int type even if it's a float
def convert_stat(x, new_value=np.nan):
    if not isinstance(x, int):

        if not isinstance(x, float) and '.' not in x:
            return new_value if x == "TRANSFER" else int(x)
        else:
            return new_value if x == "TRANSFER" else int(float(x))

    else:
        return x


def get_student_data(path, bin=False):
    dataForGraph = pd.read_csv(path)
    dataForGraph["Transferred"] = dataForGraph["A6"].apply(lambda x: True if x == "TRANSFER" else False)

    chronic_threshold = 18

    # convert absent and tardy columsn to integers
    for i in ["A", "T"]:
        for j in column_list(i, 6, 13):
            dataForGraph[j] = dataForGraph[j].apply(convert_stat)
            # check median and mean and see what happens!
            dataForGraph[j].fillna(dataForGraph[j].median(), inplace=True)

    # hronically absent at least one grade
    dataForGraph["ChronicallyAbsent_in_HS"] = (dataForGraph["A9"] >= chronic_threshold) | \
                                              (dataForGraph["A10"] >= chronic_threshold) | \
                                              (dataForGraph["A11"] >= chronic_threshold)
                                              # (dataForGraph["A12"] >= chronic_threshold)

    dataForGraph['AbsentSum'] = dataForGraph[column_list('A', 6, 13)].sum(axis=1)
    dataForGraph['TardySum'] = dataForGraph[column_list('T', 6, 13)].sum(axis=1)

    # for different time periods
    dataForGraph['AbsencesSum_MS'] = dataForGraph[column_list('A', 6, 9)].sum(axis=1)
    dataForGraph['AbsencesSum_HS'] = dataForGraph[column_list('A', 9, 13)].sum(axis=1)

    dataForGraph['TardiesSum_MS'] = dataForGraph[column_list('T', 6, 9)].sum(axis=1)
    dataForGraph['TardiesSum_HS'] = dataForGraph[column_list('T', 9, 13)].sum(axis=1)

    # Impute the reduced lunch column
    dataForGraph["Student on Free or Reduced Lunch"] = \
        dataForGraph["Student on Free or Reduced Lunch"].apply(lambda x: "No" if pd.isnull(x) else x.strip())
    print("Number of students on reduced lunch: {}".format(
        dataForGraph[dataForGraph["Student on Free or Reduced Lunch"] == "Yes"].shape[0]))

    # Impute disability column
    dataForGraph["Has a Disability?"].fillna("No", inplace=True)
    dataForGraph["Has_504"] = dataForGraph["Has a Disability?"].apply(lambda x: "Yes" if '504' in x else "No")

    # Calculate Absence Rates
    for k in ["A", "T"]:
        for column_name in column_list(k, 6, 13):
            dataForGraph[column_name + "_rate"] = dataForGraph[column_name] / 180

    # chronically absent at least one grade
    dataForGraph["ChronicallyAbsent_in_MS"] = (dataForGraph["A6"] >= chronic_threshold) | \
                                              (dataForGraph["A7"] >= chronic_threshold) | \
                                              (dataForGraph["A8"] >= chronic_threshold)

    # BINNING
    if bin:
        for i in ["A", "T"]:
            for j in column_list(i, 6, 13):
                dataForGraph[j] = dataForGraph[j].apply(lambda x: int(x / 8))

    # remove_outliers(dataForGraph, dataForGraph["AbsencesSum_HS"], 0, 0.95)
    # TODO: Check if this does anyting
    # dataForGraph.reset_index()

    return dataForGraph
