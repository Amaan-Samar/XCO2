import datetime
import matplotlib.pyplot as plt
# https://matplotlib.org/stable/gallery/style_sheets/style_sheets_reference.html
plt.style.use('seaborn')

import numpy as np
import os
import os.path
import pandas as pd

from IPython.core.display import HTML
from IPython.display import clear_output
from IPython.core.pylabtools import figsize

from oco_tools import *

file_dict = dict_oco2_xco2

path_files = 'downloads/subset_OCO2_L2_Lite_FP_11.1r_20230920_091205_'

# NZ bounding box
latMin = -48
latMax = -34
lonMin = 165
lonMax = 179


oco2_aoi = OCO2(path_files+'/*.nc4', file_dict, latMin, latMax, lonMin, lonMax)

t_unit_oco2    = "seconds since 1970-1-1 0:0:0"
t_cal = u"gregorian"

# Convert oco2 time units to a python time
oco2_aoi.time_python  = convert_time(oco2_aoi.time, t_unit_oco2, t_cal)
oco2_aoi.time_str = [datetime.datetime.fromisoformat(str(date)) for date in oco2_aoi.time_python]

# plot a timeseries
plt.figure(figsize=(20,6))
plt.subplot(1,2,1)
plt.plot(oco2_aoi.time_str, oco2_aoi.xco2,'.')
plt.ylabel('XCO2 (ppm)')
plt.title('NZ 2014 CO2 Levels (OCO-2 XCO2)')
plt.savefig('image_results/oco2-xco2-nz-2014.png')


# let us look at a histogram
plt.figure(figsize=(10,6))
plt.hist(oco2_aoi.xco2, 100,range=(400,415), density=True, alpha=0.5, label='XCO2 (ppm)')
plt.title('NZ 2014 CO2 Levels (OCO-2 XCO2)')
plt.legend(loc=0)
plt.savefig('image_results/oco2-xco2-nz-2014-hist.png')

print('Standard deviation of measurement ' + str(np.std(oco2_aoi.xco2)) )
print('Expected standard deviation based on posteriori error ' + str(np.mean(oco2_aoi.xco2_sigma)) )

# Get data every 3 days and create a running mean of +/- 3 days:
# Define data range to smooth on (every 3 days)
from datetime import datetime
dates = np.arange('2014-01', '2015-12', dtype='datetime64[3D]').astype(datetime)

# Use dates with a 3 day time-step and now use a +/- 3 day filter for the running mean:
timeseries_oco2 = running_mean(oco2_aoi.time_str, oco2_aoi.xco2, dates, 3)

plt.figure(figsize=(15,6))
plt.errorbar(dates, timeseries_oco2.mean,yerr=timeseries_oco2.standard_error, label='OCO-2 XCO2 Mean')
plt.ylabel('XCO2 (ppm)')
plt.legend(loc=0)
plt.title('NZ 2014 CO2 Levels (OCO-2 XCO2) +/-3 day running mean')
plt.savefig('image_results/oco2-xco2-nz-2014-ravg.png')