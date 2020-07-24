from sklearn.base import BaseEstimator
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsRegressor

class SumTransformer(BaseEstimator):

    def __init__(self):
        pass

    def fit(self, x, y=None):
        return self

    def transform(self, df):
        df['AbsentSum'] = df[column_list('A', 6, 13)].sum(axis=1)
        df['TardySum'] = df[column_list('T', 6, 13)].sum(axis=1)

        # for different time periods
        df['AbsencesSum_MS'] = df[column_list('A', 6, 9)].sum(axis=1)
        df['AbsencesSum_HS'] = df[column_list('A', 9, 13)].sum(axis=1)

        df['TardiesSum_MS'] = df[column_list('T', 6, 9)].sum(axis=1)
        df['TardiesSum_HS'] = df[column_list('T', 9, 13)].sum(axis=1)
