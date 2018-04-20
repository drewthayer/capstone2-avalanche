from sklearn.linear_model import LinearRegression, Lasso
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import PolynomialFeatures
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def engineer_timelag_features(data_df, lag=3, fname='aspen_data'):
    cols = data_df.columns
    nn = range(data_df.shape[0])
    mm = range(data_df.shape[1])

    lags = range(lag)
    for m in mm:
        col = data_df.iloc[:,m].values
        for i in lags:
            col_new = np.insert(col,0,0) # add a zero in first element
            col_new = np.delete(col_new,-1) # remove last element
            data_df[str(cols[m])+'_{}'.format(i+1)] = col_new
            col = col_new

    # remove lagged day features
    if lag == 1:
        data_df.drop('jday_1', axis=1, inplace=True)
        data_df.drop('month_1', axis=1, inplace=True)
    if lag == 2:
        data_df.drop('jday_1', axis=1, inplace=True)
        data_df.drop('jday_2', axis=1, inplace=True)
        data_df.drop('month_1', axis=1, inplace=True)
        data_df.drop('month_2', axis=1, inplace=True)
    if lag == 3:
        data_df.drop('jday_1', axis=1, inplace=True)
        data_df.drop('jday_2', axis=1, inplace=True)
        data_df.drop('jday_3', axis=1, inplace=True)
        data_df.drop('month_1', axis=1, inplace=True)
        data_df.drop('month_2', axis=1, inplace=True)
        data_df.drop('month_3', axis=1, inplace=True)
    if lag == 4:
        data_df.drop('jday_1', axis=1, inplace=True)
        data_df.drop('jday_2', axis=1, inplace=True)
        data_df.drop('jday_3', axis=1, inplace=True)
        data_df.drop('jday_4', axis=1, inplace=True)
        data_df.drop('month_1', axis=1, inplace=True)
        data_df.drop('month_2', axis=1, inplace=True)
        data_df.drop('month_3', axis=1, inplace=True)
        data_df.drop('month_4', axis=1, inplace=True)
    if lag == 5:
        data_df.drop('jday_1', axis=1, inplace=True)
        data_df.drop('jday_2', axis=1, inplace=True)
        data_df.drop('jday_3', axis=1, inplace=True)
        data_df.drop('jday_4', axis=1, inplace=True)
        data_df.drop('jday_5', axis=1, inplace=True)
        data_df.drop('month_1', axis=1, inplace=True)
        data_df.drop('month_2', axis=1, inplace=True)
        data_df.drop('month_3', axis=1, inplace=True)
        data_df.drop('month_4', axis=1, inplace=True)
        data_df.drop('month_5', axis=1, inplace=True)

    pickle.dump(data_df, open( fname + '_lag{}.p'.format(lag), 'wb'))
    return data_df

if __name__=='__main__':
    # load data
    data_df = pickle.load( open( 'aspen_data_less.p', 'rb'))

    df = engineer_timelag_features(data_df, lag=5, fname='aspen_data')
