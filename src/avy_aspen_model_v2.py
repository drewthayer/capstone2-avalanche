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

#from feat_importance_plot import feat_importance_plot

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

def feat_importance_plot(model,names,filename,color='g',alpha=0.5,fig_size=(10,10),dpi=250):
    '''
    horizontal bar plot of feature importances
    works for sklearn models that have a .feature_importances_ method (e.g. RandomForestRegressor)

    imputs
    ------
    model:    class:     a fitted sklearn model
    names:    list:      list of names for all features
    filename: string:    name of file to write, with appropriate path and extension (e.g. '../figs/feat_imp.png')

    optional imputs to control plot
    ---------------
    color(default='g'), alpha(default=0.8), fig_size(default=(10,10)), dpi(default=250)

    '''
    ft_imp = 100*model.feature_importances_ / np.sum(model.feature_importances_) # funny cause they sum to 1
    ft_imp_srt, ft_names, ft_idxs = zip(*sorted(zip(ft_imp, names, range(len(names)))))

    idx = np.arange(len(names))
    plt.figure(figsize=(10,10))
    plt.barh(idx, ft_imp_srt, align='center', color=color,alpha=alpha)
    plt.yticks(idx, ft_names)

    plt.title("Feature Importances in {}".format(model.__class__.__name__))
    plt.xlabel('Relative Importance of Feature', fontsize=14)
    plt.ylabel('Feature Name', fontsize=14)
    plt.tight_layout()
    plt.savefig(filename,dpi=dpi)
    plt.show()

if __name__=='__main__':
    # load data
    data_df = pickle.load( open( 'aspen_data_lag4.p', 'rb'))

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

    ''' random forest '''
    rfr = RandomForestRegressor(n_estimators = 300, n_jobs = -1, oob_score=True)
    rfr.fit(X_train, y_train)
    # metrics
    oob = rfr.oob_score_
    print('rfr out-of-bag train score = {:0.3f}'.format(oob))
    importances_rfr = rfr.feature_importances_
    rfr_feats = sorted(zip(X_train.columns, importances_rfr), key=lambda x:abs(x[1]), reverse=True)
    # predictions
    preds_rfr = rfr.predict(X_test)
    rmse = np.sqrt(np.sum((y_test - preds_rfr)**2))
    print('rfr test rmse = {:0.3f}'.format(rmse))

    # plot
    fig, ax = plt.subplots(1,1,figsize=(6,4))
    h1 = ax.plot(range(len(y_test)),y_test,'b', label='actual')
    h2 = ax.plot(range(len(preds_rfr)),preds_rfr,'orange', label='predicted - rfr')
    ax.set_ylabel('daily # of avalanches')
    ax.set_title('Aspen, CO: avalanches >= D2')
    ax.legend()
    plt.show()

    # feature importance plot
    names = X_train.columns
    filename = 'rfr_lag4_feats'
    feat_importance_plot(rfr,names,filename,color='g',alpha=0.5,fig_size=(10,10),dpi=250)

    # save output
    true = y_test.values.reshape(-1,1)
    predicted = preds_rfr.reshape(-1,1)
    output = np.concatenate((true, predicted), axis=1)
    pickle.dump(output, open('output_rfr.p', 'wb'))



    ''' gradient boosting regressor '''
    best_params = {'loss': 'lad', 'max_depth': 6, 'max_features': 'log2', 'n_estimators': 600}
    gbr = GradientBoostingRegressor(**best_params)

    gbr.fit(X_train,y_train)
    score = cross_val_score(gbr, X_train, y_train, cv=10, n_jobs=-1)
    print('gbr cval training score = {:0.3f}'.format(np.mean(score)))

    preds_gbr = gbr.predict(X_test)
    rmse = np.sqrt(np.sum((y_test - preds_gbr)**2))
    print('gbr test rmse = {:0.3f}'.format(rmse))

    #oob = gbr.oob_improvement_
    train_score = gbr.train_score_
    importances_gbr = gbr.feature_importances_

    # plot
    fig, ax = plt.subplots(1,1,figsize=(6,4))
    h1 = ax.plot(range(len(y_test)),y_test,'b', label='actual')
    h2 = ax.plot(range(len(preds_gbr)),preds_gbr,'r', label='predicted - gbr')
    ax.set_ylabel('daily # of avalanches')
    ax.set_title('Aspen, CO: avalanches >= D2')
    ax.legend()
    plt.show()

    # feature importance plot
    names = X_train.columns
    filename = 'gbr_lag4_feats'
    feat_importance_plot(gbr,names,filename,color='teal',alpha=0.5,fig_size=(10,10),dpi=250)

    # training plot
    fig, ax = plt.subplots(1,1,figsize=(6,4))
    ax.plot(train_score,'-k')
    ax.set_xlabel('boosting stage')
    ax.set_ylabel('training score')
    ax.set_title('Gradient Boosting Regressor training')

    # save output
    true = y_test.values.reshape(-1,1)
    predicted = preds_gbr.reshape(-1,1)
    output = np.concatenate((true, predicted), axis=1)
    pickle.dump(output, open('output_rgbr.p', 'wb'))

    # ''' poly splines '''
    # from sklearn.preprocessing import PolynomialFeatures
    # from sklearn.pipeline import make_pipeline
    #
    # best_params = {'loss': 'lad', 'max_depth': 6, 'max_features': 'log2', 'n_estimators': 300}
    # gbr = GradientBoostingRegressor(**best_params)
    #
    # rfr = RandomForestRegressor(n_estimators = 300, n_jobs = -1, oob_score=True)
    #
    # degree = 2
    # pipe = make_pipeline(PolynomialFeatures(degree), rfr)
    # pipe.fit(X_train, y_train)
    #
    # #train_score = gbr.train_score_
    # #importances_gbr = gbr.feature_importances_
    #
    # oob = rfr.oob_score_
    # importances_rfr = rfr.feature_importances_
    # rfr_feats = sorted(zip(X_train.columns, importances_rfr), key=lambda x:abs(x[1]), reverse=True)
    #
    #
    # #gbr_feats = sorted(zip(X_train.columns, importances_gbr), key=lambda x:abs(x[1]), reverse=True)
    #
    # # score = cross_val_score(pipe,X_train,y_train,cv=10)
    # # print('rfr poly training score = {:0.3f}'.format(np.mean(score)))
    #
    # preds_pipe = pipe.predict(X_test)
    # rmse = np.sqrt(np.sum((y_test - preds_pipe)**2))
    # print('rfr poly test rmse = {:0.3f}'.format(rmse))
    # #
    # # plot
    # fig, ax = plt.subplots(1,1,figsize=(6,4))
    # h1 = ax.plot(range(len(y_test)),y_test,'b', label='actual')
    # h2 = ax.plot(range(len(preds_pipe)),preds_pipe,'orange', label='predicted')
    # ax.set_ylabel('daily # of avalanches')
    # ax.set_title('Aspen, CO: avalanches >= D2')
    # ax.legend()
    # plt.show()
    #
