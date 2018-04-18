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


if __name__=='__main__':
    # load data
    X = pickle.load( open( "aspen_X_nosummer.p", "rb" ) )
    y = pickle.load( open( "aspen_y_nosummer.p", "rb" ) )

    # randomized ttsplit
    #X_train, X_test, y_train, y_test = train_test_split(X.values,y.values,test_size=0.2)

    # train test split in time
    splitdate = pd.to_datetime('2016-06-01')
    X_train = X[X.index <= splitdate]
    X_test = X[X.index > splitdate]
    y_train = y[y.index <= splitdate]
    y_test = y[y.index > splitdate]

    ''' linear regression '''
    linear = LinearRegression()
    linear.fit(X_train,y_train)
    score = cross_val_score(linear,X_train,y_train,cv=10)
    print('linear regression cval training score = {:0.3f}'.format(np.mean(score)))

    preds_linear = linear.predict(X_test)
    rmse = np.sqrt(np.sum((y_test - preds_linear)**2))
    print('linear regression test rmse = {:0.3f}'.format(rmse))

    linear_coefs = linear.coef_

    # Lasso regression
    lasso = Lasso(alpha=0.001)
    lasso.fit(X_train,y_train)
    score = cross_val_score(lasso,X_train,y_train,cv=10)
    print('linear L1 regression cval training score = {:0.3f}'.format(np.mean(score)))

    preds_lasso = lasso.predict(X_test)
    rmse = np.sqrt(np.sum((y_test - preds_lasso)**2))
    print('linear L1 regression test rmse = {:0.3f}'.format(rmse))

    lasso_coefs = lasso.coef_


    ''' gradient boost regressor '''
    best_params = {'loss': 'lad', 'max_depth': 6, 'max_features': 'log2', 'n_estimators': 600}
    gbr = GradientBoostingRegressor(**best_params)

    gbr.fit(X_train,y_train)
    score = cross_val_score(gbr,X_train,y_train,cv=10)
    print('gbr cval training score = {:0.3f}'.format(np.mean(score)))

    preds_gbr = gbr.predict(X_test)
    rmse = np.sqrt(np.sum((y_test - preds_gbr)**2))
    print('gbr test rmse = {:0.3f}'.format(rmse))

    #oob = gbr.oob_improvement_
    train_score = gbr.train_score_
    importances_gbr = gbr.feature_importances_

    ''' random forest regressor '''
    #gbr = GradientBoostingRegressor(loss='ls',n_estimators=500)
    # params from grid search:
    rfr = RandomForestRegressor(n_estimators = 300)
    rfr.fit(X_train,y_train)
    score = cross_val_score(rfr,X_train,y_train,cv=10)
    print('rfr cval training score = {:0.3f}'.format(np.mean(score)))

    preds_rfr = rfr.predict(X_test)
    rmse = np.sqrt(np.sum((y_test - preds_rfr)**2))
    print('rfr test rmse = {:0.3f}'.format(rmse))

    importances_rfr = rfr.feature_importances_

    # feature importances
    linear_feats = sorted(zip(X.columns, linear_coefs), key=lambda x:abs(x[1]), reverse=True)
    lasso_feats = sorted(zip(X.columns, lasso_coefs), key=lambda x:abs(x[1]), reverse=True)
    gbr_feats = sorted(zip(X.columns, importances_gbr), key=lambda x:abs(x[1]), reverse=True)
    rfr_feats = sorted(zip(X.columns, importances_rfr), key=lambda x:abs(x[1]), reverse=True)

    # figures
    fig, ax = plt.subplots(1,1,figsize=(6,4))
    ax.plot(y)
    ax.set_ylabel('daily # of avalanches')
    ax.set_title('Aspen, CO: avalanches >= D2')

    fig, ax = plt.subplots(1,1,figsize=(6,4))
    h1 = ax.plot(range(len(y_test)),y_test,'b', label='actual')
    h2 = ax.plot(range(len(preds_gbr)),preds_gbr,'orange', label='predicted')
    ax.set_ylabel('daily # of avalanches')
    ax.set_title('Aspen, CO: avalanches >= D2')
    ax.legend()
    plt.show()

    fig, ax = plt.subplots(1,1,figsize=(6,4))
    ax.plot(train_score,'-k')
    ax.set_xlabel('boosting stage')
    ax.set_ylabel('training score')
    ax.set_title('Gradient Boosting Regressor training')
    # # grid search for gradient boosting regressor
    # param_grid = {
    #     'loss': ['ls', 'lad', 'huber', 'quantile'],
    #     'n_estimators': [200,300,400,500,600],
    #     'max_depth': [2,4,6,8],
    #     'max_features': ['auto','sqrt','log2']
    #     }
    # gbr_grid = GridSearchCV(gbr,param_grid, n_jobs=-1, verbose=1)
    # gbr_grid.fit(X_train, y_train)
    #
    # best = gbr_grid.best_estimator_
    # best_params = gbr_grid.best_params_
