# example
import os
import requests
import csv
import pandas as pd
##from urllib2 import urlopen
#from datetime import datetime, timedelta

# API key
import os
api_key = os.environ['WUnderground_API_KEY']

endpoint = "http://api.wunderground.com/api/"
yyyymmdd = 20060406
STATE = 'CO'
location = 'Aspen'

#http://api.wunderground.com/api/fecc581d61db3a3a/history_20060405/q/CA/San_Francisco.json

url = ''.join([endpoint, api_key, '/history_', str(yyyymmdd), '/q/', STATE, '/', location, '.json'])
response = requests.get(url)
print('response code = {}'.format(response.status_code))
response = response.json()
print(len(response))

if len(response) > 0:
# unpack response dictionary
    metadata = list(response.values())[0]
    results = list(response.values())[1]

    # print column names to see what they are
    print(results.keys())

    # write to csv
    #csv_columns = list(results.keys())
    #csv_columns = ['mindate', 'maxdate', 'latitude', 'name', 'datacoverage', 'id', 'longitude','elevation', 'elevationUnit']
    ''' write daily summary'''
    daily_summary = results['dailysummary'][0]
    csv_columns = list(daily_summary.keys())
    fname = 'test1.csv'
    # deal with datetime
    date_dict = daily_summary['date']
    date_string = "{y}{m}{d}".format(y=date_dict['year'], m=date_dict['mon'], d=date_dict['mday'])
    # cols and values
    datacols = ['minwspdm','maxwspdm','meanwindspm','precipm']
    colnames = ['date','min_wspd','max_wspd','mean_wspd','precip']
    data = [date_string, daily_summary[datacols[0]], daily_summary[datacols[1]]]


    with open(fname, 'a') as csvfile: # 'wa for append'
                #writer = csv.DictWriter(csvfile, fieldnames=colnames)
                txtwriter = csv.writer(csvfile, delimiter=',')
                #txtwriter.writeheader()
                #txtwriter.writerow(colnames)
                txtwriter.writerow(data)
else:
    print('response object = empty')

# for day in range(4,-1,-1):
#     url = ''.join(['http://api.wunderground.com/api/', api_key,
#     '/history_',
#     (now-timedelta(days=day)).strftime('%Y%m%d'),
#     '/q/TX/Addison.json'])
#     data = download_json(url)
#     for k in data['history']['observations']:
#         y0 = float(k['pressurem'])
#         if y0 < 0.0:
#             continue
#         else:
#             x.append(x1 + float(k['date']['hour'])+
#             round((float(k['date']['min'])/60.0),2))
#             y.append(y0)
#             x1 += 24.0
#
#
# payload = {
#     'datasetid': 'GSOM',
# #    'stationid': 'COOP:010008',
# #    'units': 'metric',
# #    'locationid': 'GHCND:USC00051071',
#     'startdate': '2010-05-01',
#     'enddate': '2010-05-31',
#     'limit':1000,
# }
#
# # 2 ways to do it, with in long URL string or with endpoint, headers, params
# response = requests.get(endpoint, headers=headers, params=payload)
# print('response code = {}'.format(response.status_code))
# response = response.json()
#
# ''' '''
# def download_json(url):
#     weather = urlopen(url)
#     string = weather.read()
#     weather.close()
#     return loads(string)
#
# d,x,y=[],[],[]
# x1 = 0.0
# key = 'Your key here'
# for day in range(4,-1,-1):
#     url = ''.join(['http://api.wunderground.com/api/', key,
#     '/history_',
#     (now-timedelta(days=day)).strftime('%Y%m%d'),
#     '/q/TX/Addison.json'])
#     data = download_json(url)
#     for k in data['history']['observations']:
#         y0 = float(k['pressurem'])
#         if y0 < 0.0:
#             continue
#         else:
#             x.append(x1 + float(k['date']['hour'])+
#             round((float(k['date']['min'])/60.0),2))
#             y.append(y0)
#             x1 += 24.0
