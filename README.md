Readme
## Empirical avalanche prediction in Colorado:
#### Can a machine-learning model trained on historical climatic and avalanche data augment prediction of avalanche risk?

### Data:
__CAIC Avalanche data__

10 backcountry zones:
<img alt="caic zones" src="/pub_figs/CAIC_zones.png" width='300'>

avalanche observation data back to 1980:
<img alt="caic zones" src="/pub_figs/caic_example.png" width='500'>

__weather data__
SNOTEL sensor network:
<img alt="snotel network" src="/pub_figs/co_swe_current.png" width='500'>

Local Climatalogical Data (often airports)
https://www.ncdc.noaa.gov/cdo-web/datatools/lcd


### EDA/ data trends:
__locations:__
<img alt="avy by location" src="/figs/2018_d3_landmarks.png" width='500'>

__backcountry zones:__
BC Zone
Northern San Juan        2998
Front Range              1565
Vail & Summit County     1337
Aspen                    1210
Gunnison                 1188
Sawatch Range             806
Southern San Juan         585
Steamboat & Flat Tops     186
Grand Mesa                155
Sangre de Cristo           22

__snow angle:__ (this is well understood science)
<img alt="avy by location" src="/figs/2018_snow_angle.png" width='300'>

### preliminary modeling:
__Data:__  
 - features: wind data from Aspen airport, airtemp and precip data from Independence Pass SNOTEL
 - target: Aspen Zone avalanches, # per day (size >= D2)
<img alt='timeseries' src='figs/aspen_avys_d2plus.png' width='500'>

__model:__
 - train/test split: June 2015
 - linear regression cval training score = -0.013
 - linear regression test rmse = 16.840
 - linear L1 regression cval training score = -0.025
 - linear L1 regression test rmse = 16.871
 - gbr cval training score = -0.138
 - gbr test rmse = 19.079
<img alt="first model" src="/figs/aspen_avys_preds.png" width='500'>

#### improvements:
 - __more data!__ models need a longer data record (and more zones) to train
__more flexible models__: hard to capture the highly variable nature of a stochastic natural process

__remove summer, add jday__:
linear regression cval training score = -0.126
linear regression test rmse = 16.848

linear L1 regression cval training score = -0.120
linear L1 regression test rmse = 16.853

gbr cval training score = -0.285
gbr test rmse = 18.193 # THIS IS BROKEN

rfr cval training score = -0.749
rfr test rmse = 16.231

__best gbr model:__
linear regression cval training score = -0.126
linear regression test rmse = 16.848

linear L1 regression cval training score = -0.120
linear L1 regression test rmse = 16.853

gbr cval training score = -0.129
gbr test rmse = 16.683

rfr cval training score = -0.788
rfr test rmse = 16.492

model:
<img alt="first model"
 src="/figs/nosummer/aspen_nosummer_preds_gbr_best.png" width='500'>

model training:
<img alt="first model"
 src="/figs/nosummer/gbr_training.png" width='500'>

__aspen and leadville airports__
linear regression cval training score = -0.142
linear regression test rmse = 16.895
linear L1 regression cval training score = -0.136
linear L1 regression test rmse = 16.905
gbr cval training score = -0.105
gbr test rmse = 17.239
rfr cval training score = -0.682
rfr test rmse = 16.441


 __Oversample low-probability avalanches__
class imbalance problem :
<img alt="less features"
 src="/figs/oversample/hist_numperday.png" width='300'>
 frequency of avys/day:
 n = 2004
 In [27]: counts
 Out[27]: {0: 1533, 1: 381, 2: 42, 3: 20, 4: 7, 5: 12, 6: 8}

 balance to 1:
 n = 3977 if balanced to 1
 In [29]: factors
 Out[29]: {1: 1, 2: 9, 3: 19, 4: 54, 5: 31, 6: 47}

 balance to 0:
 n = 11522
 In [25]: factors
 Out[25]: {0: 1, 1: 4, 2: 36, 3: 76, 4: 219, 5: 127, 6: 191}

__less features? worse__

__features removed after PCA__
aspen and leadville DAILYAverageWindSpeed

linear L1 regression cval training score = -13.414
linear L1 regression test rmse = 48.547
gbr cval training score = -7.013
gbr test rmse = 20.061
rfr cval training score = -3.602
rfr test rmse = 19.435

<img alt="less features"
 src="/figs/oversample/gbr_less_from_pca.png" width='500'>

__multivariate adaptive regression splines__
https://contrib.scikit-learn.org/py-earth/content.html#api
<img alt="less features"
 src="/figs/oversample/multi_adapt_splines.png" width='500'>

 Forward Pass
---------------------------------------------------------------
iter  parent  var  knot  mse       terms  gcv    rsq    grsq   
---------------------------------------------------------------
0     -       -    -     4.433884  1      4.435  0.000  0.000  
1     0       4    1365  4.282448  3      4.287  0.034  0.033  
2     0       10   -1    4.194507  4      4.200  0.054  0.053  
3     0       9    2145  4.157366  6      4.167  0.062  0.060  
4     0       8    -1    4.139280  7      4.150  0.066  0.064  
5     0       2    -1    4.124325  8      4.137  0.070  0.067  
6     0       0    264   4.108657  10     4.124  0.073  0.070  
7     0       1    -1    4.069557  11     4.087  0.082  0.078  
8     0       5    -1    4.056928  12     4.076  0.085  0.081  
9     0       6    -1    3.978787  13     3.999  0.103  0.098  
10    0       7    -1    3.970884  14     3.993  0.104  0.100  
11    0       6    977   3.968389  16     3.993  0.105  0.100  
---------------------------------------------------------------
Stopping Condition 2: Improvement below threshold

Pruning Pass
--------------------------------------------
iter  bf  terms  mse   gcv    rsq    grsq   
--------------------------------------------
0     -   16     3.97  3.992  0.105  0.100  
1     5   15     3.97  3.991  0.105  0.100  
2     6   14     3.96  3.985  0.106  0.101  
3     8   13     3.97  3.987  0.105  0.101  
4     12  12     3.97  3.990  0.104  0.100  
5     14  11     3.97  3.990  0.104  0.100  
6     13  10     3.98  3.998  0.102  0.099  
7     4   9      3.99  4.007  0.099  0.096  
8     1   8      4.01  4.021  0.096  0.093  
9     7   7      4.04  4.048  0.090  0.087  
10    9   6      4.08  4.094  0.079  0.077  
11    10  5      4.10  4.105  0.076  0.074  
12    15  4      4.18  4.187  0.057  0.056  
13    11  3      4.21  4.215  0.050  0.050  
14    3   2      4.30  4.298  0.031  0.031  
15    2   1      4.43  4.435  0.000  0.000  
--------------------------------------------
Selected iteration: 2

Earth Model
---------------------------------------------------
Basis Function               Pruned  Coefficient   
---------------------------------------------------
(Intercept)                  No      -5.22782e+12  
h(swe_start_m-0.6223)        No      -379.922      
h(0.6223-swe_start_m)        No      -5.06174      
month                        No      0.089386      
h(precip_incr_m-0.02286)     No      -519.329      
h(0.02286-precip_incr_m)     Yes     None          
precip_start_m               Yes     None          
leadville_Daily_peak_wind    No      -0.0176392    
h(aspen_Daily_peak_wind-52)  No      -0.16687      
h(52-aspen_Daily_peak_wind)  No      -0.108398     
aspen_SustainedWindSpeed     No      -0.169434     
airtemp_max_C                No      -0.0657349    
airtemp_min_C                No      4.70504e+12   
airtemp_mean_C               No      -0.090332     
h(airtemp_min_C-1.11111)     No      -4.70504e+12  
h(1.11111-airtemp_min_C)     No      4.70504e+12   
---------------------------------------------------
MSE: 3.9639, GCV: 3.9855, RSQ: 0.1060, GRSQ: 0.1013

__polynomial spline the features:__

finally starting to capture some of the behavior
<img alt="less features"
 src="/figs/oversample/gbr_spline3.png" width='500'>

 poly 2:
 gbr poly training score = -6.985
gbr poly test rmse = 19.931

 poly 3:
 gbr poly training score = -6.455
 gbr poly test rmse = 19.597

 poly 4:
 gbr poly training score = -6.507
 gbr poly test rmse = 19.742

poly3:
 [('aspen_SustainedWindSpeed', 0.01668406401900523),
 ('airtemp_mean_C', 0.012609223746917182),
 ('swe_start_m', 0.012543717909469961),
 ('airtemp_max_C', 0.011110875872774692),
 ('leadville_SustainedWindSpeed', 0.011024357196882415),
 ('leadville_Daily_peak_wind', 0.010865338688127155),
 ('precip_start_m', 0.0095716385022798327),
 ('airtemp_min_C', 0.0075883984821047542),
 ('month', 0.0073560504187418106),
 ('precip_incr_m', 0.0071991492254596971),
 ('jday', 0.0022579295906576956),
 ('aspen_Daily_peak_wind', 0.0)]

 poly4:
 [('leadville_SustainedWindSpeed', 0.0014734698962991925),
 ('aspen_SustainedWindSpeed', 0.0014308879524915765),
 ('precip_start_m', 0.00096387341271134944),
 ('airtemp_max_C', 0.00086604904616177257),
 ('precip_incr_m', 0.00067493655975634219),
 ('airtemp_mean_C', 0.00066136450592446888),
 ('airtemp_min_C', 0.00056780713597166972),
 ('leadville_Daily_peak_wind', 0.00049914960504373641),
 ('jday', 9.7089286087595852e-05),
 ('swe_start_m', 5.2432338695591005e-05),
 ('month', 1.987035141430264e-06),
 ('aspen_Daily_peak_wind', 0.0)]

 __feature engineer timeseries__
next step: incorporate time_series info over a 3-day window
feature engineer features for d-1, d-2, and d-3

linear L1 regression cval training score = -11.206
linear L1 regression test rmse = 41.282

gbr cval training score = -0.217
gbr test rmse = 20.737

rfr cval training score = 0.369
rfr test rmse = 17.444

_gradient boost_
<img alt="less features"
 src="/figs/timelag/gbr_lag3.png" width='500'>

_random forest_
 <img alt="less features"
  src="/figs/timelag/rfr_lag3.png" width='500'>

__lagged with poly splines__
poly 2:
gbr poly training score = -0.594
gbr poly test rmse = 19.292

_gradient boost with lag3, poly2_
 <img alt="less features"
  src="/figs/timelag/gbr_lag3_poly2.png" width='500'>
