import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter

df = pd.read_csv('data/CAIC_avalanches_2017-11-01_2018-04-10.csv')
'''
columns: ['Obs ID', 'Date', 'Date Known', 'Time', 'Time Known', 'BC Zone',
       'HW Zone', 'Operation', 'Landmark', 'First Name', 'Last Name',
       'HW Path', '#', 'Elev', 'Asp', 'Type', 'Trigger', 'Trigger_sub',
       'Rsize', 'Dsize', 'Incident', 'Area Description', 'Comments',
       'Avg Slope Angle', 'Start Zone Elev', 'Start Zone Elev units',
       'Sliding Sfc', 'Weak Layer', 'Weak Layer Type', 'Avg Width',
       'Max Width', 'Width units', 'Avg Vertical', 'Max Vertical',
       'Vertical units', 'Avg Crown Height', 'Max Crown Height',
       'Crown Height units', 'Terminus ']
'''
# where are d3 avalanches?
d3 = df[df.Dsize == 'D3']
d4 = df[df.Dsize == 'D4'] # none in 2018

# when do d3s happen?
c = Counter(d3.Date)
ts,counts = zip(*c.items())
plt.plot(ts,counts,'ob')
plt.gcf().autofmt_xdate()
plt.title('2018 D3 avalanches')
plt.ylabel('# of avalanches')
plt.savefig('figs/2018_d3_datetime.png',dpi=250)
plt.close()

# where do d3s happen?
landmark = d3.Landmark[d3.Landmark.notnull()]
c_landmark = Counter(landmark)
landmarks,counts = zip(*c_landmark.items())
plt.bar(landmarks,counts)
plt.gcf().autofmt_xdate()
plt.title('2018 D3 avalanches')
plt.savefig('figs/2018_d3_landmarks.png',dpi=250)
plt.close()
