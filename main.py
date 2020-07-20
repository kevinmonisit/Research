import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns

dataForGraph = pd.read_csv('data/High School East Student Data - Sheet1.csv')


# easy way of accessing A_6, A_7, ... A_N columns
def column_list(letter, start, end):
    return ["%s%d" % (letter, i) for i in range(start, end)]


# convert strings to int type even if it's a float
def convert_stat(x):
    if not isinstance(x, int):

        # we don't know if the string is float or int
        # converting it to int if it's not an int will cause an error
        try:
            return 0 if x == "TRANSFER" else int(x)
        except:
            pass

        # if it can't pass as an int, then it must be a float that will be converted to an int
        # the float is rounded
        return 0 if x == "TRANSFER" else int(float(x))

    else:
        return x


# convert absent and tardy columsn to integers
for i in ["A", "T"]:
    for j in column_list(i, 6, 13):
        dataForGraph[j] = dataForGraph[j].apply(convert_stat)

print(list(dataForGraph.iloc[0].values[5:12]))  # absence columns of first student
print(list(dataForGraph.iloc[0].values[12:19]))  # tardy columns
