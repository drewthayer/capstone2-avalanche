import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
import os
import sys
sys.path.insert(0, 'src')
#sys.path.append('/src')
from clean_snow_data import clean_snow_data
import pickle

def clean_airport_data(df,name):
    airport_df = df
    cols = ['STATION','ELEVATION','DATE','DAILYDeptFromNormalAverageTemp',
       'DAILYAverageRelativeHumidity', 'DAILYAverageDewPointTemp',
       'DAILYAverageWetBulbTemp',
       'DAILYPrecip', 'DAILYSnowfall', 'DAILYSnowDepth',
       'DAILYAverageStationPressure', 'DAILYAverageSeaLevelPressure',
       'DAILYAverageWindSpeed', 'DAILYPeakWindSpeed', 'PeakWindDirection',
       'DAILYSustainedWindSpeed', 'DAILYSustainedWindDirection']
    airport_df = airport_df[cols]
    # get only daily data
    airport_df = airport_df[~np.isnan(airport_df['DAILYAverageWindSpeed'])]
    airport_df['day'] = airport_df.DATE.str.rsplit(' ',n=1).str[0]
    airport_df['datetime'] = pd.to_datetime(airport_df.day)
    airport_df.set_index(airport_df.datetime, inplace=True)

    # clean up columns the easy way (mixed types, some loss) DAILYSustainedWindSpeed, DAILYSustainedWindDirection
    airport_df['SustainedWindDirection'] = airport_df['DAILYSustainedWindDirection'].convert_objects(convert_numeric=True)
    airport_df['SustainedWindSpeed'] = airport_df['DAILYSustainedWindSpeed'].convert_objects(convert_numeric=True)
    airport_df['DeptFromNormalAvgTemp'] = airport_df['DAILYDeptFromNormalAverageTemp'].convert_objects(convert_numeric=True)
    airport_df['Precip'] = airport_df['DAILYPrecip'].convert_objects(convert_numeric=True)
    airport_df['Daily_peak_wind'] = airport_df['DAILYPeakWindSpeed'].convert_objects(convert_numeric=True)
    airport_df['Peak_wind_direction'] = airport_df['PeakWindDirection'].convert_objects(convert_numeric=True)

    # add 'name' to col tags
    save_cols = ['DeptFromNormalAvgTemp', 'DAILYAverageRelativeHumidity',
    'DAILYAverageDewPointTemp', 'DAILYAverageWetBulbTemp','DAILYAverageWindSpeed',
    'Daily_peak_wind', 'Peak_wind_direction', 'SustainedWindSpeed',
    'SustainedWindDirection']

    save_cols_less = ['DAILYAverageWindSpeed',
    'Daily_peak_wind', 'SustainedWindSpeed',]

    save_cols_labels = []
    for label in save_cols_less:
        save_cols_labels.append(''.join([name, '_', label]))

    airport_df = airport_df[save_cols_less]
    airport_df.columns = save_cols_labels

    return(airport_df)
# import data cleaning scripts
#
if __name__=='__main__':
    # paths
    current = os.getcwd()
    caic_dir = os.path.dirname(''.join([current,'/../data-caic/']))
    noaa_dir = os.path.dirname(''.join([current,'/../data-noaa/']))
    snotel_dir = os.path.dirname(''.join([current,'/../data-snotel/']))

    # LCD wind speed data
    fname1 = noaa_dir + '/LCD_data/' + 'aspen_pitkin_airport_20060101_current.csv'
    airport_df1 = pd.read_csv(fname1)

    fname2 = noaa_dir + '/LCD_data/' + 'leadville_lake_airport_20090101.csv'
    airport_df2 = pd.read_csv(fname2)

    airport_aspen = clean_airport_data(airport_df1, 'aspen')
    airport_leadville = clean_airport_data(airport_df2, 'leadville')


    # avalanche data
    fname = caic_dir + '/' + 'CAIC_avalanches_2010-11-01_2018-04-10.csv'
    avy_df = pd.read_csv(fname)
    avy_df['datetime'] = pd.to_datetime(avy_df.Date)
    avy_df.set_index(avy_df.datetime, inplace=True)

    avy_aspen = avy_df[avy_df['BC Zone'] == 'Aspen']
    ''' df size: 10128 x 40 '''

    # snotel_data
    stationnames = ['618_mcclure_pass','669_north_lost_trail','737_schofield_pass','542_independence_pass',
                    '369_brumley','547_ivanhoe']
    data_list = []
    for stationname in stationnames:
        snow_df = pd.read_csv(snotel_dir + '/' + 'snotel_{}.csv'.format(stationname),header=58)
        snow_df.columns = ['dt_string', 'swe_start_in',
       'precip_start_in',
       'airtemp_max_F', 'airtemp_min_F',
       'airtemp_mean_F', 'precip_incr_in']
        snow_df = clean_snow_data(snow_df)
        data_list.append(snow_df)

    ''' merge data '''
    # just independence pass
    snow_df0 = data_list[3]
    merge_airport = pd.merge(airport_aspen, airport_leadville, how='left', left_index=True, right_index=True)
    merge1 = pd.merge(merge_airport, snow_df0, how='left', left_index=True, right_index=True)
    merge2 = pd.merge(merge1, avy_aspen, how='left', left_index=True, right_index=True)

    ''' month and julian day feature '''
    merge2['dt'] = merge2.index
    merge2['jday'] = merge2.dt.apply(lambda x: x.timetuple().tm_yday)

    ''' option: only D2+ avalanches'''
    merge2['D2_up'] = np.where(np.in1d(merge2.Dsize, ['D2','D2.5','D3','D3.5','D4']), merge2['#'], 0)

    ''' option: remove summer '''
    merge2['month'] = merge2.index.month
    nosummer = merge2[(merge2.month < 6) | (merge2.month >= 11)]

    ''' clip at start date of avy record '''
    startday = '2010-11-15'
    clipped = nosummer[nosummer.index > pd.to_datetime(startday)]

    ''' select feature columns'''
    # lists of target columns
    asp = [x for x in airport_aspen.columns]
    led = [x for x in airport_leadville.columns]
    sno = ['swe_start_m', 'airtemp_max_C', 'airtemp_min_C',
    'airtemp_mean_C', 'precip_start_m',
    'precip_incr_m']
    other = ['month','jday']
    # combine and select columns
    feature_list = asp + led + sno + other
    data_df = clipped[feature_list]
    # impute nans
    data_df = data_df.fillna(0)

    ''' unsupervised '''
    D2 = clipped[clipped.Dsize == 'D2']
    D2 = D2[feature_list]
    D2 = D2.fillna(0)
    D3 = clipped[clipped.Dsize == 'D3']
    D3 = D3[feature_list]
    D3 = D3.fillna(0)

    from sklearn.decomposition import PCA
    pca = PCA(n_components=2, copy=True, svd_solver='auto', iterated_power='auto', random_state=None)
    # D2
    pca.fit(D2)
    var = pca.explained_variance_
    var_ratio = pca.explained_variance_ratio_
    components = pca.components_
    # D3
    pca.fit(D3)
    var = pca.explained_variance_
    var_ratio = pca.explained_variance_ratio_
    components = pca.components_


    #barplots of components
    fig,ax = plt.subplots(1,2,figsize=(8,4))
    plt.title('Aspen: D2 avalanches')
    ax[0].bar(D3.columns,components[0])
    ax[0].set_title('PC1: explained variance = {:0.3f}'.format(var_ratio[0]))
    plt.xticks(rotation=90)
    ax[1].bar(D3.columns,components[1])
    ax[1].set_title('PC2: explained variance = {:0.3f}'.format(var_ratio[1]))
    #fig.autofmt_xdate()
    plt.setp( ax[0].xaxis.get_majorticklabels(), rotation=90 )
    plt.setp( ax[1].xaxis.get_majorticklabels(), rotation=90 )
    plt.tight_layout()
    #plt.xticks(rotation=90)








    ''' save dataframe '''
    #pickle.dump( data_df, open( "aspen_data_less.p", "wb" ) )


    ''' QA / QC '''
    # X.dtypes
    # mask = X.applymap(np.isreal).sum()

    ''' EDA '''
    counts = avy_df.groupby(['BC Zone']).size().sort_values(ascending=False)
