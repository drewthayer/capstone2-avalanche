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

<img alt="snotel network" src="/pub_figs/nrcs_snotel_eyak_ak.jpg" width='200'>

_source: NRCS National Water and Climate Center, USDA_

Local Climatalogical Data (commonly airports):
<img alt="snotel network" src="/pub_figs/airport_weather_station.jpg" width='200'>

### EDA/ data trends:
__destructive size:__
<img alt="avy by location" src="/figs/dsize.png" width='300'>

_this modeling approach will consider avalanches D2 or greater_

__D2+ avalanches by backcountry zone:__
- Northern San Juan        2998
- Front Range              1565
- Vail & Summit County     1337
- Aspen                    1210
- Gunnison                 1188
- Sawatch Range             806
- Southern San Juan         585
- Steamboat & Flat Tops     186
- Grand Mesa                155
- Sangre de Cristo           22

__snow angle:__ (this is well understood science)
<img alt="avy by location" src="/figs/2018_snow_angle.png" width='300'>

### modeling strategy:

<img alt="caic zones" src="/pub_figs/aspen_closeup.png" width='200'>

__Data:__  
 - _features:_ wind data from Aspen and Leadville airports, air temperature and precipitation data from Independence Pass SNOTEL station
 - _target:_ Aspen Zone avalanches, # per day (size >= D2)
 - _train and test split:_ June 2016

<img alt='timeseries' src='figs/aspen_avys_d2plus.png' width='500'>

__improvements to feature matrix:__
 - remove summer months (no avalanches possible)
 - add day and month (seasonal dependency)

__preliminary linear model:__
 - linear L1 regression cval training score = -0.025
 - linear L1 regression test rmse = 16.871

<img alt="first model" src="/figs/aspen_lasso.png" width='500'>

 - avalanches are stochastic phenomena dependent on non-linear processes
 - linear model can 'guess low' every time, high penalty for departure from mean
 - need a more flexible model


__preliminary gradient boosting regression model:__

 - gbr cval training score = -0.129
 - gbr test rmse = 16.683

 |model             |  training |
 |:-------------------------:|:-------------------------:|
 |![](figs/nosummer/aspen_nosummer_preds_gbr_best.png)  |  ![](figs/nosummer/gbr_training.png)|

 - starting to capture non-linear behavior
 - takes many boosting stages (600+) to train the model:


 __Addressing the class imbalance problem:__
 -  classes are highly imbalanced

<img alt="less features"
 src="/figs/oversample/hist_numperday.png" width='300'>

 frequency of avalanches/day:
 - {0: 1533, 1: 381, 2: 42, 3: 20, 4: 7, 5: 12, 6: 8}

 factors to balance classes:
  - {0: 1, 1: 4, 2: 36, 3: 76, 4: 219, 5: 127, 6: 191}

class-balanced gbr model:

<img alt="less features"
 src="/figs/oversample/gbr_spline3.png" width='500'>
 _moving in the right direction..._
 - stochastic behavior: can now predict values > 2
 - still not a good fit

__other experiments:__
 - PCA to remove features: inconclusive
 - polynomial features to incorporate time-smoothed information: poor performance


 __incorporate time-series information: feature engineering__
 - processes that create avalanches are highly time-sequence dependent
 - strategy: engineer features for past days over time window
   - for day $i$ in $n$ days: 
   - for every feature $X_{i}$ add  $X_{(i-1)}$, $X_{(i-2)}$, $X_{(i-3)}$, ...

gradient boost, lag = 3 days:
model             |  training
|:-------------------------:|:-------------------------:|
|![](figs/timelag/gbr_lag3.png)  |  ![](figs/timelag/gbr_lag4_train.png)|


random forest lag, = 3 days:
model             |  |
|:-------------------------:|:-------------------------:|
|![](figs/timelag/rfr_lag3.png)  |  |

 _now we're getting somewhere..._
  - much better accuracy
  - training faster

 process: test models with time lags from 1 to 5 days


## model selection 1: train/test performance
- features: balanced classes, 4-day time-lagged features

random forest regressor            |  gradient boosting regressor
:-------------------------:|:-------------------------:
|   out-of-bag training score = 0.992 | cross-validated training score = -0.217 |
|  test RMSE = 16.392 | test RMSE = 20.737 |
![](figs/timelag/rfr_lag4_label.png)  |  ![](figs/timelag/gbr_lag4_label.png)

## model selection 2: use of feature space
random forest regressor             |  gradient boosting regressor
:-------------------------:|:-------------------------:
![](figs/timelag/rfr_lag4_feats.png)  |  ![](figs/timelag/gbr_lag4_feats.png)

 - random forest model can be trained with fewer features

## model selection 3: receiver operating characteristic
__the goal:__ predict the risk of avalanches
 - a predicted __risk__ is more useful than a predicted #


__classify predictions:__ ordinal --> binary
 - select threshold
 - if number of avalanches >= threshold, 1, else 0
 - compare predictions and true record of events

__Receiver Operating Characteristic:__
 <img alt="less features"
   src="/figs/model_metrics/ROC_rfr_gbr_t1.png" width='600'>

 - ROC compares True Positive rate to False Positive rate
 - for risk prediction:
   - false positives are OK
   - true positives must be maximized
   - false negatives must be penalized (danger zone)
     - maximize recall

__selected model random forest regressor:__
  - better ROC metrics (higher accuracy and recall)
  - can be trained with fewer features
  - doesn't predict negative values

__next step:__ choose a threshold to maximize recall

## decisions for model implementation: accuracy, precision, recall
_prediction range goes up to 6..._

<img alt="less features"
  src="/figs/model_metrics/acc_rec_prec_rfr.png" width='400'>

_...but performance hard to interpret at predictions >= 1_

__limit prediction range between 0 and 1:__
<img alt="less features"
  src="/figs/model_metrics/acc_rec_prec_rfr_t1.png" width='600'>

__most accurate and precise model:__ threshold = 0.75
 - best if your goal is to see an avalanche
 - accuracy = 0.782
 - precision = 0.632
 - recall= 0.381

|          |predicted 0| predicted 1|
|----------|----------|-----------|
|__actual 0__   |       298|        70|
|__actual 1__   |        25|         43|



__balanced model with high recall:__ threshold = 0.46
 - most conservative model for risk forecasting
 - recall= 0.735
 - accuracy = 0.713
 - precision = 0.466

|          |predicted 0| predicted 1|
|----------|----------|-----------|
|__actual 0__   |       228|        30|
|__actual 1__   |        95|         83|

### improvements:
 - __more data!__ models need a longer data record (and more backcountry zones) to train
__more flexible models__: hard to capture the highly variable nature of a stochastic natural process
