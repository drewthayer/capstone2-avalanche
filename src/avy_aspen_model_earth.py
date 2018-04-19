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

def oversample(data_df):

    ''' oversample days with many avalanches '''
    # plt.hist(target)
    # plt.title('frequency distribution: # of D2+ avalanches per day')
    # plt.ylabel('frequency')

    # frequency of avys/day:
    # n = 2004
    #  {0: 1533, 1: 381, 2: 42, 3: 20, 4: 7, 5: 12, 6: 8, 7: 0}

    avy0 = data_df[data_df.D2_up == 0]
    avy1 = data_df[data_df.D2_up == 1]
    avy2 = data_df[data_df.D2_up == 2]
    avy3 = data_df[data_df.D2_up == 3]
    avy4 = data_df[data_df.D2_up == 4]
    avy5 = data_df[data_df.D2_up == 5]
    avy6 = data_df[data_df.D2_up == 6]

    n_avy = [0,1,2,3,4,5,6]
    counts = {}
    for n in n_avy:
        counts[n] = data_df[data_df.D2_up == n].count().max()

    # duplication factors
    # {2: 9, 3: 19, 4: 54, 5: 31, 6: 47}
    factors = {}
    for n in n_avy: # no 7 b/c 0
        factors[n] = counts[0]//counts[n]

    # concatenate to dataframe
    mini_dfs = [avy0, avy1, avy2, avy3, avy4, avy5, avy6]
    frames = []
    for n in factors.keys():
        i = 0
        while i <= factors[n]:
            frames.append(mini_dfs[n])
            i += 1

    df = pd.concat(frames, axis=0)

    # random shuffle
    df_shuffle = df.copy()
    df_shuffle.set_index(np.random.permutation(df_shuffle.index), inplace=True)

    return df, df_shuffle
    #df_shuffle.sort_index(inplace=True)


if __name__=='__main__':
    # load data
    data_df = pickle.load( open( 'aspen_data_lag3.p', 'rb'))

    # randomized ttsplit
    #X_train, X_test, y_train, y_test = train_test_split(X.values,y.values,test_size=0.2)

    # train test split in time
    splitdate = pd.to_datetime('2016-06-01')
    train_df = data_df[data_df.index <= splitdate]
    test_df = data_df[data_df.index > splitdate]

    # oversample days with avalanches to balance classes
    unshuffled, train_shuffle = oversample(train_df)

    ''' select features and target X,y '''
    # define target and remove target nans
    ycol = 'D2_up'
    # train set
    X_train = train_shuffle.copy()
    y_train = X_train.pop(ycol)
    # test set
    X_test = test_df.copy()
    y_test = X_test.pop(ycol)

    ''' earth model '''
    # #Fit an Earth model
    # from pyearth import Earth
    # model = Earth(penalty=2, minspan=4, smooth=False)
    # model.fit(X_train,y_train)
    #
    # #Print the model
    # print(model.trace())
    # print(model.summary())
    #
    # score = cross_val_score(model,X_train,y_train,cv=10)
    # print('earth cval training score = {:0.3f}'.format(np.mean(score)))
    #
    # preds_earth = model.predict(X_test)
    # rmse = np.sqrt(np.sum((y_test - preds_earth)**2))
    # print('linear regression test rmse = {:0.3f}'.format(rmse))

    ''' poly splines '''
    from sklearn.preprocessing import PolynomialFeatures
    from sklearn.pipeline import make_pipeline

    best_params = {'loss': 'lad', 'max_depth': 6, 'max_features': 'log2', 'n_estimators': 300}
    gbr = GradientBoostingRegressor(**best_params)

    rfr = RandomForestRegressor(n_estimators = 300, n_jobs = -1, oob_score=True)

    degree = 2
    pipe = make_pipeline(PolynomialFeatures(degree), rfr)
    pipe.fit(X_train, y_train)

    train_score = gbr.train_score_
    importances_gbr = gbr.feature_importances_

    oob = rfr.oob_score_
    importances_rfr = rfr.feature_importances_


    gbr_feats = sorted(zip(X_train.columns, importances_gbr), key=lambda x:abs(x[1]), reverse=True)

    score = cross_val_score(pipe,X_train,y_train,cv=10)
    print('rfr poly training score = {:0.3f}'.format(np.mean(score)))

    preds_pipe = pipe.predict(X_test)
    rmse = np.sqrt(np.sum((y_test - preds_pipe)**2))
    print('rfr poly test rmse = {:0.3f}'.format(rmse))

    # plot
    fig, ax = plt.subplots(1,1,figsize=(6,4))
    h1 = ax.plot(range(len(y_test)),y_test,'b', label='actual')
    h2 = ax.plot(range(len(preds_pipe)),preds_pipe,'orange', label='predicted')
    ax.set_ylabel('daily # of avalanches')
    ax.set_title('Aspen, CO: avalanches >= D2')
    ax.legend()
    plt.show()

    fig, ax = plt.subplots(1,1,figsize=(6,4))
    ax.plot(train_score,'-k')
    ax.set_xlabel('boosting stage')
    ax.set_ylabel('training score')
    ax.set_title('Gradient Boosting Regressor training')


    # ''' linear regression '''
    # linear = LinearRegression()
    # linear.fit(X_train,y_train)
    # score = cross_val_score(linear,X_train,y_train,cv=10)
    # print('linear regression cval training score = {:0.3f}'.format(np.mean(score)))
    #
    # preds_linear = linear.predict(X_test)
    # rmse = np.sqrt(np.sum((y_test - preds_linear)**2))
    # print('linear regression test rmse = {:0.3f}'.format(rmse))
    #
    # linear_coefs = linear.coef_
    #
    # # Lasso regression
    # lasso = Lasso(alpha=0.001)
    # lasso.fit(X_train,y_train)
    # score = cross_val_score(lasso, X_train, y_train, cv=10, n_jobs=-1)
    # print('linear L1 regression cval training score = {:0.3f}'.format(np.mean(score)))
    #
    # preds_lasso = lasso.predict(X_test)
    # rmse = np.sqrt(np.sum((y_test - preds_lasso)**2))
    # print('linear L1 regression test rmse = {:0.3f}'.format(rmse))
    #
    # lasso_coefs = lasso.coef_
    #
    #
    # ''' gradient boost regressor '''
    # best_params = {'loss': 'lad', 'max_depth': 6, 'max_features': 'log2', 'n_estimators': 600}
    # gbr = GradientBoostingRegressor(**best_params)
    #
    # gbr.fit(X_train,y_train)
    # score = cross_val_score(gbr, X_train, y_train, cv=10, n_jobs=-1)
    # print('gbr cval training score = {:0.3f}'.format(np.mean(score)))
    #
    # preds_gbr = gbr.predict(X_test)
    # rmse = np.sqrt(np.sum((y_test - preds_gbr)**2))
    # print('gbr test rmse = {:0.3f}'.format(rmse))
    #

    # train_score = gbr.train_score_
    # importances_gbr = gbr.feature_importances_
    #
    # ''' random forest regressor '''
    # #gbr = GradientBoostingRegressor(loss='ls',n_estimators=500)
    # # params from grid search:
    # rfr = RandomForestRegressor(n_estimators = 300, n_jobs = -1, oob_score=True)
    # rfr.fit(X_train,y_train)
    # score = cross_val_score(rfr, X_train, y_train, cv=10, n_jobs=-1)
    # print('rfr cval training score = {:0.3f}'.format(np.mean(score)))
    #
    # preds_rfr = rfr.predict(X_test)
    # rmse = np.sqrt(np.sum((y_test - preds_rfr)**2))
    # print('rfr test rmse = {:0.3f}'.format(rmse))
    #
    # importances_rfr = rfr.feature_importances_
    # oob = rfr.oob_score_
    #
    # # feature importances
    # linear_feats = sorted(zip(X_train.columns, linear_coefs), key=lambda x:abs(x[1]), reverse=True)
    # lasso_feats = sorted(zip(X_train.columns, lasso_coefs), key=lambda x:abs(x[1]), reverse=True)
    # gbr_feats = sorted(zip(X_train.columns, importances_gbr), key=lambda x:abs(x[1]), reverse=True)
    # rfr_feats = sorted(zip(X_train.columns, importances_rfr), key=lambda x:abs(x[1]), reverse=True)

    # figures
    # fig, ax = plt.subplots(1,1,figsize=(6,4))
    # ax.plot(y_train,'ob')
    # ax.set_ylabel('daily # of avalanches')
    # ax.set_title('Aspen, CO: avalanches >= D2')
