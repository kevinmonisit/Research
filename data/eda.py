import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer

# dataForGraph = pd.read_csv('../input/dataproj/High School East Student Data - Sheet1.csv')
dataForGraph = pd.read_csv('data.csv')


def remove_outliers(data, target, first, second):
    non_outliers = target.between(target.quantile(first), target.quantile(second))
    count = 0
    for index in range(0, len(data)):
        if ~non_outliers[index]:
            count += 1
            data.drop(index, inplace=True)

    print("{} outliers removed".format(count))


# easy way of accessing A_6, A_7, ... A_N columns
def column_list(letter, start, end):
    return ["%s%d" % (letter, i) for i in range(start, end)]


# convert strings to int type even if it's a float
def convertStat(x, new_value=np.nan):
    if not isinstance(x, int):

        if not isinstance(x, float) and '.' not in x:
            return new_value if x == "TRANSFER" else int(x)
        else:
            return new_value if x == "TRANSFER" else int(float(x))

    else:
        return x


number_of_transfer_students = dataForGraph[dataForGraph["A6"] == "TRANSFER"].shape

dataForGraph["Transferred"] = dataForGraph["A6"].apply(lambda x: True if x == "TRANSFER" else False)

# convert absent and tardy columsn to integers
for i in ["A", "T"]:
    for j in column_list(i, 6, 13):
        imputer = SimpleImputer()
        dataForGraph[j] = dataForGraph[j].apply(convertStat)
        dataForGraph[j].fillna(dataForGraph[j].mean(), inplace=True)

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

remove_outliers(dataForGraph, dataForGraph["AbsencesSum_HS"], 0, 0.95)
dataForGraph.reset_index()

# Impute the reduced lunch column
dataForGraph["Student on Free or Reduced Lunch"] = dataForGraph["Student on Free or Reduced Lunch"].apply(
    lambda x: "No" if pd.isnull(x) else x.strip())
print("Number of students on reduced lunch: {}".format(
    dataForGraph[dataForGraph["Student on Free or Reduced Lunch"] == "Yes"].shape[0]))

# Impute disability column
dataForGraph["Has a Disability?"].fillna("No", inplace=True)
dataForGraph["Has_504"] = dataForGraph["Has a Disability?"].apply(lambda x: "Yes" if '504' in x else "No")

print("run")

#==================


# returns the number of chronically absent students that meet a certain condition
def get_n_chronically_absent(column_condition):
    return dataForGraph.loc[(dataForGraph["ChronicallyAbsent_in_HS"] == True) & \
                            (dataForGraph[column_condition] != "No")].shape[0]


number_of_chronically_absent = dataForGraph.loc[(dataForGraph["ChronicallyAbsent_in_HS"] == True)].shape[0]

# number of chronically absent students that meet a certain condition
chronic_absent_columns = {
    "iep": get_n_chronically_absent("IEP/Specialized"),
    "reduced_lunch": get_n_chronically_absent("Student on Free or Reduced Lunch"),
    '504': get_n_chronically_absent("Has_504")
}

print("Number of chronically absent students in sample: ", number_of_chronically_absent)

print(chronic_absent_columns)
numberOfChronicAbsent = dataForGraph[dataForGraph["ChronicallyAbsent_in_HS"] == True]

columns = numberOfChronicAbsent.columns.values
print(columns)
# print(numberOfChronicAbsent[["Student on Free or Reduced Lunch", "ChronicallyAbsent_in_HS"]])

students_test = dataForGraph.loc[(dataForGraph["ChronicallyAbsent_in_HS"] == True) &
                                 (dataForGraph["Has_504"] != "No") &
                                 (dataForGraph["Student on Free or Reduced Lunch"] != "No") &
                                 (dataForGraph["IEP/Specialized"] != "No")]
print(students_test)
# chronic_absent_df = pd.DataFrame(chronic_absent_columns.items(), column=chronic_absent_columns)
