import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import sys
sys.path.insert(0, 'src')
#sys.path.append('/src')
from clean_snow_data import clean_snow_data

# import data cleaning scripts
#
if __name__=='__main__':
    df = pd.read_csv('data/CAIC_avalanches_2010-11-01_2018-04-10.csv')
    df['datetime'] = pd.to_datetime(df.Date)
    df.set_index(df.datetime, inplace=True)
    d2 = df[df.Dsize == 'D2']
    d3 = df[df.Dsize == 'D3']
    ''' df size: 10128 x 40 '''

    stationnames = ['335_berthoud','680_park_cone','762_slumgullion','701_porphyry_creek','737_schofield_pass',
                '682_park_reservoir','669_north_lost_trail','1059_cochetopa_pass','618_mcclure_pass',
                '409_columbine_pass','675_overland_reservoir','622_mesa_lakes','538_idarado',
               '713_red_mtn_pass']
    data_list = []
    for stationname in stationnames:
        snow_df = pd.read_csv('snotel-data/snotel_{}.csv'.format(stationname),header=58)
        snow_df.columns = ['dt_string', 'swe_start_in',
       'precip_start_in',
       'airtemp_max_F', 'airtemp_min_F',
       'airtemp_mean_F', 'precip_incr_in']

        snow_df = clean_snow_data(snow_df)
        data_list.append(snow_df)

    df0 = data_list[0]
    merged_df = pd.merge(df, df0, how='left', left_index=True, right_index=True)

    # predict d3 avalanches
    merged_df['d3'] = np.where(merged_df.Dsize == 'D3', 1, 0)
    X = merged_df[[ 'swe_start_m',
       'airtemp_max_C', 'airtemp_min_C', 'airtemp_mean_C', 'precip_start_m',
       'precip_incr_m']]
    y = merged_df.d3
    X = X.fillna(0)

    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score

    logistic = LogisticRegression()
    logistic.fit(X.values,y.values)
    score = cross_val_score(logistic,X.values,y.values,scoring='accuracy',cv=10)

    ''' In [21]: score
Out[21]:
array([ 0.95167653,  0.95167653,  0.95162883,  0.95162883,  0.95162883,
        0.95162883,  0.95256917,  0.95256917,  0.95256917,  0.95256917])

In [25]: coefs (exp)
Out[25]:
array([[ 1.69852928,  0.98855897,  1.01084818,  0.97375582,  2.28315312,
         1.02886805]])
         '''

    # precip at d3 avalanches
    #d3 = df[merged_df.Dsize == 'D3']

    d3['d3'] = np.where(d3.Dsize == 'D3', 1, 0)
    counts_d3 = d3.groupby(d3.index.date).count()['d3']
    d3['counts'] = counts_d3
    merged_d3 = pd.merge(d3, df0, how='left', left_index=True, right_index=True)

    # precip and avy
    prcp = merged_d3.precip_incr_m
    plt.plot(prcp,merged_d3.counts,'ob')
    plt.savefig('freq_d3_precip_1station.png',dpi=250)

    # do for avy > D2

    d_avy = df[np.in1d(df.Dsize,['D2','D2.5','D3'])]
    d_avy['avy'] = 1
    counts_avy = d_avy.groupby(d_avy.index.date).count()['avy']
    d_avy['counts'] = counts_avy
    merged_avy = pd.merge(d_avy, df0, how='left', left_index=True, right_index=True)

    # precip and avy
    prcp = merged_avy.precip_incr_m
    plt.plot(prcp,merged_avy.counts,'ob')
    plt.savefig('freq_d2_sxd3_precip_1station.png',dpi=250)

    # predict d2,d3 avalanches
    #merged_df['d3'] = np.where(merged_df.Dsize == 'D3', 1, 0)
    merged_avy['d3'] = np.where(merged_avy.Dsize == 'D3', 1, 0)
    X = merged_avy[['month','swe_start_m',
       'airtemp_max_C', 'airtemp_min_C', 'airtemp_mean_C', 'precip_start_m',
       'precip_incr_m','d3']]
    X = X.dropna()
    y = X.pop('d3')
    #y = merged_avy.d3

    X_train, X_test, y_train, y_test = train_test_split(X.values,y.values,test_size=0.2)



    #linear = LinearRegression()
    #linear.fit(X.values,y.values)
    logistic = LogisticRegression()
    logistic.fit(X_train,y_train)
    score = cross_val_score(logistic,X_train,y_train,scoring='accuracy',cv=10)
    print('cval training score = {:0.3f}'.format(np.mean(score)))

    preds = logistic.predict(X_test)
    misclass = np.sum((y_test - preds))/np.sum(y_test)
    print('test misclassification rate = {:0.3f}'.format(misclass))

    ''' cval training score = 0.907, with nans = 0, Xshape = 5247
        cval training score = 0.908, with nans removed, Xshape = 4781
        test misclassification rate = 1.000

        whoops.... misbalanced classes, logistic model is just predicting 0
        In [21]: np.sum(y_train)/len(y_train)
        Out[21]: 0.0938
        In [20]: np.sum(y_test)/len(y_test)
        Out[20]: 0.0856
    '''

    ''' linear regression: counts'''
    X = merged_avy[['month','swe_start_m',
       'airtemp_max_C', 'airtemp_min_C', 'airtemp_mean_C', 'precip_start_m',
       'precip_incr_m','counts']]
    X = X.dropna()
    y = X.pop('counts')

    X_train, X_test, y_train, y_test = train_test_split(X.values,y.values,test_size=0.1)

    linear = LinearRegression()
    linear.fit(X.values,y.values)
    score = cross_val_score(linear,X_train,y_train,cv=10)
    print('regression cval training score = {:0.3f}'.format(np.mean(score)))

    preds = linear.predict(X_test)
    rmse = np.sqrt(np.sum((y_test - preds)**2))
    print('regression test rmse = {:0.3f}'.format(rmse))

    ''' regression is pretty bad...
        regression cval training score = 0.126
        regression test rmse = 484.124
    '''

    ''' gradient boost regressor '''
    from sklearn.ensemble import GradientBoostingRegressor
    gbr = GradientBoostingRegressor(loss='ls',n_estimators=500)
    gbr.fit(X_train,y_train)
    score = cross_val_score(gbr,X_train,y_train,cv=10)
    print('gbr cval training score = {:0.3f}'.format(np.mean(score)))

    preds = gbr.predict(X_test)
    rmse = np.sqrt(np.sum((y_test - preds)**2))
    print('gbr test rmse = {:0.3f}'.format(rmse))

    importances = gbr.feature_importances_
    '''
    GBR:
    with 20 percent holdout
    gbr cval training score = 0.964
    gbr test rmse = 102.345
    [('month', 0.04664721446905238),
     ('swe_start_m', 0.2281261802536606),
     ('airtemp_max_C', 0.15791222685970527),
     ('airtemp_min_C', 0.13829045912707968),
     ('airtemp_mean_C', 0.121289305148257),
     ('precip_start_m', 0.23925971926472675),
     ('precip_incr_m', 0.068474894877518247)]

     with 10 percent holdout
     gbr cval training score = 0.965
     gbr test rmse = 64.751
     [('month', 0.04679122063582862),
     ('swe_start_m', 0.23774231874216759),
     ('airtemp_max_C', 0.15780226210104306),
     ('airtemp_min_C', 0.14516317061173911),
     ('airtemp_mean_C', 0.10927508839258848),
     ('precip_start_m', 0.23873472151343936),
     ('precip_incr_m', 0.064491218003193587)]
    '''
