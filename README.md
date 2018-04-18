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

### EDA/ data trends:
__locations:__
<img alt="avy by location" src="/figs/2018_d3_landmarks.png" width='500'>
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

 - feature importances:
   - _linear regression:_
   [('precip_incr_m', -1.4239286312343558),
    ('swe_start_m', 0.39096482892964407),
    ('precip_start_m', 0.22421629912945473),
    ('SustainedWindSpeed', -0.011763454061623293),
    ('DeptFromNormalAvgTemp', 0.011085732824780826),
    ('Daily_peak_wind', 0.0106426281068223),
    ('airtemp_mean_C', -0.010350017249809017),
    ('airtemp_min_C', -0.0072678204646204909),
    ('airtemp_max_C', -0.0068153847130960482),
    ('DAILYAverageRelativeHumidity', 0.0025001391964069274),
    ('DAILYAverageWetBulbTemp', -0.0015979443666795877),
    ('DAILYAverageDewPointTemp', -0.0013336135672252347),
    ('DAILYAverageWindSpeed', -0.00043554913619548666),
    ('SustainedWindDirection', -0.00041163006794109336),
    ('Peak_wind_direction', 0.00012344237923546386)]

   - _gradient boosting regressor:_
   [('precip_start_m', 0.82688645138191375),
    ('DeptFromNormalAvgTemp', 0.12305347648274112),
    ('Daily_peak_wind', 0.031885084117312917),
    ('airtemp_min_C', 0.0047823262512151458),
    ('precip_incr_m', 0.0023003185582740975),
    ('airtemp_max_C', 0.0018598320258386329),
    ('DAILYAverageDewPointTemp', 0.001810889077790248),
    ('SustainedWindSpeed', 0.0017931830793473668),
    ('DAILYAverageWetBulbTemp', 0.001713003181693478),
    ('swe_start_m', 0.0015172313894999377),
    ('DAILYAverageRelativeHumidity', 0.001223573701209627),
    ('airtemp_mean_C', 0.0011746307531612418),
    ('DAILYAverageWindSpeed', 0.0),
    ('Peak_wind_direction', 0.0),
    ('SustainedWindDirection', 0.0)]

   - _lasso regression:_
   [('airtemp_min_C', -0.016055909323589117),
 ('DeptFromNormalAvgTemp', 0.0062194939407182615),
 ('DAILYAverageWetBulbTemp', -0.0026114274183084685),
 ('Daily_peak_wind', 0.0025632029862054789),
 ('DAILYAverageRelativeHumidity', 0.0023044873595614258),
 ('DAILYAverageDewPointTemp', -0.0006354680306766042),
 ('SustainedWindDirection', -0.00036316601148379357),
 ('Peak_wind_direction', 0.00017741649447880725),
 ('DAILYAverageWindSpeed', 0.0),
 ('SustainedWindSpeed', 0.0),
 ('swe_start_m', 0.0),
 ('airtemp_max_C', 0.0),
 ('airtemp_mean_C', -0.0),
 ('precip_start_m', 0.0),
 ('precip_incr_m', 0.0)]

wet bulb temp: The wet-bulb temperature is the lowest temperature which may be achieved by evaporative cooling of a water-wetted (or even ice-covered), ventilated surface.

#### improvements:
 - __more data!__ models need a longer data record (and more zones) to train
__more flexible models__: hard to capture the highly variable nature of a stochastic natural process
