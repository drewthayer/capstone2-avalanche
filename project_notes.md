## project notes

#### trial run: aspen area avalanches

first shot: target all types of avalanche
 - linear regression cval training score = 0.149
 - linear regression test rmse = 24.462
 - gbr cval training score = 0.079
 - gbr test rmse = 26.238

second try: target d2+ avalanches, with data back to 2010. does not generalize well
 - linear regression cval training score = 0.068
 - linear regression test rmse = 16.073
 - gbr cval training score = 0.018
 - gbr test rmse = 16.954

 importances:
 [('DeptFromNormalAvgTemp', 0.11978404237041888),
 ('DAILYAverageRelativeHumidity', 0.048565977044967235),
 ('DAILYAverageDewPointTemp', 0.045574632362183079),
 ('DAILYAverageWetBulbTemp', 0.03572161505796588),
 ('DAILYAverageWindSpeed', 0.061375010795957877),
 ('Daily_peak_wind', 0.054591884418552616),
 ('Peak_wind_direction', 0.072801894414867091),
 ('SustainedWindSpeed', 0.026184685608727758),
 ('SustainedWindDirection', 0.044671584278345965),
 ('swe_start_m', 0.11398825537558716),
 ('airtemp_max_C', 0.09793839723883313),
 ('airtemp_min_C', 0.074977062723142571),
 ('airtemp_mean_C', 0.090418618588753705),
 ('precip_start_m', 0.089007411230590402),
 ('precip_incr_m', 0.024398928491106652)]

third try: split in time on june 2015
 - linear regression cval training score = -1.904
 - linear regression test rmse = 20.031
 - gbr cval training score = -6.214
 - gbr test rmse = 21.140

gbr importances:
 [('precip_incr_m', 0.026526900584930537),
 ('DAILYAverageWetBulbTemp', 0.027503878069285431),
 ('DAILYAverageDewPointTemp', 0.036182270615462718),
 ('Peak_wind_direction', 0.040082108807196246),
 ('SustainedWindSpeed', 0.045370848242963314),
 ('SustainedWindDirection', 0.052520078896964237),
 ('DAILYAverageRelativeHumidity', 0.055824613358230579),
 ('Daily_peak_wind', 0.05849201683518708),
 ('airtemp_min_C', 0.073120502458027661),
 ('airtemp_mean_C', 0.084075563701200298),
 ('precip_start_m', 0.084803056533648599),
 ('DAILYAverageWindSpeed', 0.086730149744726148),
 ('swe_start_m', 0.093080892524305445),
 ('airtemp_max_C', 0.11121922388778364),
 ('DeptFromNormalAvgTemp', 0.12446789574008806)]

 linear importances:
 [('precip_incr_m', -0.66681605155033841),
 ('SustainedWindSpeed', -0.014538021269635382),
 ('airtemp_mean_C', -0.014394905096042618),
 ('airtemp_min_C', -0.0077754709360570952),
 ('airtemp_max_C', -0.0030854636318061943),
 ('DAILYAverageWetBulbTemp', -0.0025250071999227556),
 ('DAILYAverageDewPointTemp', -0.00034003347966550523),
 ('SustainedWindDirection', -0.00027740228132754063),
 ('Peak_wind_direction', -6.6369769346014129e-05),
 ('DAILYAverageRelativeHumidity', 0.0026975522137147132),
 ('DeptFromNormalAvgTemp', 0.007154210100321),
 ('DAILYAverageWindSpeed', 0.01145721862355617),
 ('Daily_peak_wind', 0.011959475537466919),
 ('precip_start_m', 0.076510073490707708),
 ('swe_start_m', 0.47596112426887011)]

try 4: split on july 2016
linear regression cval training score = -0.013
linear regression test rmse = 16.565
gbr cval training score = -0.428
gbr test rmse = 17.092

In [118]: linear_feats
Out[118]:
[('precip_incr_m', -0.66681605155033841),
 ('SustainedWindSpeed', -0.014538021269635382),
 ('airtemp_mean_C', -0.014394905096042618),
 ('airtemp_min_C', -0.0077754709360570952),
 ('airtemp_max_C', -0.0030854636318061943),
 ('DAILYAverageWetBulbTemp', -0.0025250071999227556),
 ('DAILYAverageDewPointTemp', -0.00034003347966550523),
 ('SustainedWindDirection', -0.00027740228132754063),
 ('Peak_wind_direction', -6.6369769346014129e-05),
 ('DAILYAverageRelativeHumidity', 0.0026975522137147132),
 ('DeptFromNormalAvgTemp', 0.007154210100321),
 ('DAILYAverageWindSpeed', 0.01145721862355617),
 ('Daily_peak_wind', 0.011959475537466919),
 ('precip_start_m', 0.076510073490707708),
 ('swe_start_m', 0.47596112426887011)]

In [119]: gbr_feats
Out[119]:
[('SustainedWindSpeed', 0.025759644577647296),
 ('precip_incr_m', 0.027705877185550409),
 ('DAILYAverageWetBulbTemp', 0.029270292793761185),
 ('DAILYAverageDewPointTemp', 0.029467975565703592),
 ('DAILYAverageRelativeHumidity', 0.045106842193087103),
 ('Peak_wind_direction', 0.045986792067671596),
 ('SustainedWindDirection', 0.050020049873581718),
 ('DAILYAverageWindSpeed', 0.0737931442074167),
 ('airtemp_min_C', 0.077949534933821465),
 ('Daily_peak_wind', 0.08059652240813453),
 ('precip_start_m', 0.080902053633978982),
 ('airtemp_mean_C', 0.085877840821368137),
 ('airtemp_max_C', 0.096752876557626502),
 ('swe_start_m', 0.10443138233551821),
 ('DeptFromNormalAvgTemp', 0.14637917084513252)]

simple grid search:
 GradientBoostingRegressor(alpha=0.9, criterion='friedman_mse', init=None,
             learning_rate=0.1, loss='lad', max_depth=3, max_features=None,
             max_leaf_nodes=None, min_impurity_decrease=0.0,
             min_impurity_split=None, min_samples_leaf=1,
             min_samples_split=2, min_weight_fraction_leaf=0.0,
             n_estimators=300, presort='auto', random_state=None,
             subsample=1.0, verbose=0, warm_start=False)

In [123]: best_params
Out[123]: {'loss': 'lad', 'max_depth': 3, 'n_estimators': 300}
gbr test rmse = 19.079
