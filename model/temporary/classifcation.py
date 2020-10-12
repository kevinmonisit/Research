import copy as cp

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
# dataForGraph = pd.read_csv('../input/dataproj/High School East Student Data - Sheet1.csv')
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

dataForGraph = pd.read_csv('../../data/data.csv')

# Here i was looking for the outliers and who they were. Fix it up later.

def remove_outliers(data, target, first, second):
    non_outliers = target.between(target.quantile(first), target.quantile(second))
    outliers = pd.DataFrame()

    outliers_indices = non_outliers[non_outliers == False]
    #outliers = data[[2, 3]].copy()
    for index in range(0, len(outliers_indices)):
        outliers = outliers.append(data.iloc[outliers_indices.index[index]])

    #  print(non_outliers[non_outliers is False].index[0])
  #  print(non_outliers[non_outliers is False].index[1])
    count = 0
    index = 0
"""
    while index < data.shape[0]:
        if ~non_outliers[index]:
            count += 1
            index -= 1
            outliers.append(data.iloc[index])
            data.drop(index, inplace=True)

        index += 1
"""
  #  print("{} outliers removed".format(count))


# easy way of accessing A_6, A_7, ... A_N columns
def column_list(letter, start, end):
    return ["%s%d" % (letter, k) for k in range(start, end)]


# # convert strings to int type even if it's a float
# def convert_stat(x, new_value=np.nan):
#     if not isinstance(x, int):
#
#         if not isinstance(x, float) and '.' not in x:
#             return new_value if x == "TRANSFER" else int(x)
#         else:
#             return new_value if x == "TRANSFER" else int(float(x))
#
#     else:
#         return x


def convert_stats(x, new_value=np.nan, bins=1):
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


number_of_transfer_students = dataForGraph[dataForGraph["A6"] == "TRANSFER"].shape

dataForGraph["Transferred"] = dataForGraph["A6"].apply(lambda x: True if x == "TRANSFER" else False)

# convert absent and tardy columsn to integers
bins_used_when_preprocessing = 8
for i in ["A", "T"]:
    for j in column_list(i, 6, 13):
        imputer = SimpleImputer()
        dataForGraph[j] = dataForGraph[j].apply(convert_stats, args=(np.nan, bins_used_when_preprocessing))
        # check median and mean and see what happens!
        dataForGraph[j].fillna(dataForGraph[j].median(), inplace=True)

# chronically absent at least one grade
dataForGraph["ChronicallyAbsent_in_HS"] = (dataForGraph["A9"] >= 18) | \
                                          (dataForGraph["A10"] >= 18) | \
                                          (dataForGraph["A11"] >= 18) | \
                                          (dataForGraph["A12"] >= 18)

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

dataForGraphWithOutliers = cp.deepcopy(dataForGraph)
#remove_outliers(dataForGraph, dataForGraph["AbsencesSum_HS"], 0, 0.95)
dataForGraph.reset_index()

print("run")

#====================================== CLASSIFICATION


from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB


#features = ["Has a Disability?", "Has_504", "Student on Free or Reduced Lunch", "A6", "A7", "A8",
#                "IEP/Specialized"]

#features = ["A6", "A7", "A8", "Gender", "Has_504", "Student on Free or Reduced Lunch",
#            "IEP/Specialized"]

features = ["A8", "A7", "A6", "T6", "T7", "T8", "Has_504", "Student on Free or Reduced Lunch",
            "IEP/Specialized"]

pre_process = ColumnTransformer(remainder='passthrough',
                                transformers=[('categories',
                                              OneHotEncoder(handle_unknown="error"),
                                              ["Has_504",
                                               "Student on Free or Reduced Lunch",
                                               "IEP/Specialized"])])


# pre_process = ColumnTransformer(remainder='passthrough',
#                                 transformers=[('imputer', SimpleImputer(strategy='median',
#                                                                         fill_value=0,
#                                                                         missing_values=np.nan)),
#                                               ('categories',
#                                                OneHotEncoder(),
#                                                ["Has_504",
#                                                 "Student on Free or Reduced Lunch",
#                                                 "IEP/Specialized"])])
#

#pre_process = ColumnTransformer(remainder='passthrough',
#                                transformers=[('categories',
#                                               OneHotEncoder(),
#                                               ["Gender", "IEP/Specialized"])])
#features = ["A6", "A7", "A8", "Gender", "IEP/Specialized"]

model_pipeline = Pipeline(steps=[('transform', pre_process),
                                 ('model', GaussianNB())])

# explanation: a selected group might not have certain features, which doesn't create certain columns
X_train, X_test, y_train, y_test = train_test_split(dataForGraph[features], dataForGraph['ChronicallyAbsent_in_HS'],
                                                    test_size=0.2, random_state=1)
# pre_process.fit(X_train)
# X_train = pre_process.transform(X_train)
# X_test = pre_process.transform(X_test)
# # print(X_train)
X_train = pre_process.fit_transform(X_train)
X_test = pre_process.fit_transform(X_test)
print(X_train)
#model_pipeline.fit(X_train, y_train)

#predicted = model_pipeline.predict(X_test)

from tpot import TPOTClassifier
# from dask.distributed import Client
#
# #client = Client(n_workers=4, threads_per_worker=1)

pipeline = TPOTClassifier(generations=250,
                          population_size=30,
                          cv=4,
                          random_state=1,
                          verbosity=2,
                          n_jobs=-1,
                          warm_start=True,
                          )
# print(client)
print(X_test.shape)
print(X_train.shape)
pipeline.fit(X_train, y_train)

print(pipeline.score(X_test, y_test))
#pipeline.export("tpot_pipeline.py")
