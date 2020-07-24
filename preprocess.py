import pandas as pd


# easy way of accessing A_6, A_7, ... A_N columns
def column_list(letter, start, end):
    return ["%s%d" % (letter, i) for i in range(start, end)]


# convert strings to int type even if it's a float
# replace by median or mean?
def convert_stat(x):
    if not isinstance(x, int):

        if not isinstance(x, float) and '.' not in x:
            return 0 if x == "TRANSFER" else int(x)
        else:
            return 0 if x == "TRANSFER" else int(float(x))

    else:
        return x


# for pipeline, statistics, and graphs
def preprocess_student_data(df):
    df['AbsentSum'] = df[column_list('A', 6, 13)].sum(axis=1)
    df['TardySum'] = df[column_list('T', 6, 13)].sum(axis=1)

    # for different time periods
    df['AbsencesSum_MS'] = df[column_list('A', 6, 9)].sum(axis=1)
    df['AbsencesSum_HS'] = df[column_list('A', 9, 13)].sum(axis=1)

    df['TardiesSum_MS'] = df[column_list('T', 6, 9)].sum(axis=1)
    df['TardiesSum_HS'] = df[column_list('T', 9, 13)].sum(axis=1)
