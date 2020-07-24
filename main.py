import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeRegressor
pd.options.mode.chained_assignment = None  # default='warn'

# easy way of accessing A_6, A_7, ... A_N columns
def column_list(letter, start, end):
    return ["%s%d" % (letter, i) for i in range(start, end)]


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
    def __init__(self, student_data, new_value=0):
        self.new_value = new_value
        self.student_data = student_data

    def fit(self, x, y=None):
        return self

    def transform(self, df):
        # change to for i in ["A", "T"] to include tardies if need be
        for i in ["A"]:

            # corrects the data frame that will be used in the training models
            for j in column_list(i, 6, 9):
                df[j] = df[j].apply(convert_stat, self.new_value)

            # corrects the data frame not used in the training models
            # but contains the sum of absences in high school, a feature not included in
            # the data frame that is used in the models
            for j in column_list(i, 9, 13):
                student_data[j] = student_data[j].apply(convert_stat, self.new_value)

        # creates the required objective feature
        # before this statement, the column is empty
        df['AbsencesSum_HS'] = student_data[column_list('A', 9, 13)].sum(axis=1)

        return df


student_data = pd.read_csv("data/High School East Student Data - Sheet1.csv")
features = ["A6", "A7", "A8"]
student_data["AbsencesSum_HS"] = 0

pre_process = ColumnTransformer(remainder='passthrough',
                                transformers=[('categories', LabelEncoder(), ["Gender", "IEP/Specialized"])])

model_pipeline = Pipeline(steps=[('number_fix', SumTransformer(student_data)),
                               #  ('pre_process', pre_process),
                                 ('Decision Tree', DecisionTreeRegressor(max_leaf_nodes=12))
                                 ])

for j in column_list("A", 9, 13):
    student_data[j] = student_data[j].apply(convert_stat)

student_data["AbsencesSum_HS"] = student_data[column_list('A', 9, 13)].sum(axis=1)
scores = -1 * cross_val_score(model_pipeline,
                              student_data[features],
                              student_data["AbsencesSum_HS"],
                              cv=4,
                              scoring='neg_mean_absolute_error')
print(student_data["AbsencesSum_HS"])
print(scores)

