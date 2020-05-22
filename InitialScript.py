# Author: Ioannis Matzakos | Date: 01/05/2020

import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import statsmodels.api as sm
import itertools
import warnings
from pylab import rcParams

import time
from Utilities import Log

# Configure logger
log = Log.setup_logger("main")

plt.style.use('fivethirtyeight')
warnings.filterwarnings("ignore")

# Configure matplotlib
matplotlib.rcParams['axes.labelsize'] = 14
matplotlib.rcParams['xtick.labelsize'] = 12
matplotlib.rcParams['ytick.labelsize'] = 12
matplotlib.rcParams['text.color'] = 'k'

log.info("\n\n\t\t\t\t---- PROJECT: Predictions of Covid-19 in Greece ----"
         "\n\t\t\t\t----------------------------------------------------\n")

# Start calculating execution time
start_time = time.time()

log.info("Start of Execution\n\n\t\t\t\t------------ DATA MINING PIPELINE: START ------------\n")

# Load time series data
log.info("Loading Time Series Data")
confirmed = pd.read_csv("Data/time_series_covid19_confirmed_greece.csv")
log.info(f"Covid-19 confirmed cases in Greece \n{confirmed}")

deaths = pd.read_csv("Data/time_series_covid19_deaths_greece.csv")
log.info(f"Deaths due to Covid-19 in Greece \n{deaths}")

recovered = pd.read_csv("Data/time_series_covid19_recovered_greece.csv")
log.info(f"Covid-19 recovered cases in Greece \n{recovered}")

# TODO: Perform Preliminary Analysis

# TODO: Remove unwanted columns from initial time series dataset

# Transpose the three pandas data frames in order columns to become rows
# and establish two columns: Date and Number of Cases, in the next phase
log.info("Transposing the Data Frames")

confirmed_cases = confirmed.transpose()
confirmed_cases.index.names = ['Date']
confirmed_cases.rename(columns={0:'Confirmed Cases'}, inplace=True)
log.info(f"Covid-19 confirmed cases in Greece column names \n{confirmed_cases.columns}")
log.info(f"Covid-19 confirmed cases in Greece \n{confirmed_cases}")

death_cases = deaths.transpose()
death_cases.index.names = ['Date']
death_cases.rename(columns={0:'Deaths'}, inplace=True)
log.info(f"Deaths due to Covid-19 in Greece \n{death_cases}")

recovered_cases = recovered.transpose()
recovered_cases.index.names = ['Date']
recovered_cases.rename(columns={0:'Recovered Cases'}, inplace=True)
log.info(f"Covid-19 recovered cases in Greece \n{recovered_cases}")

# Merge the three transposed data frames into one in order to have the following
# columns: Date, Confirmed Cases, Deaths and Recovered Cases
log.info("Merging Data Frames")

df = confirmed_cases.merge(death_cases, left_index=True, right_index=True)
df = df.merge(recovered_cases, left_index=True, right_index=True)
# pd.set_option("display.max_rows", None, "display.max_columns", None)
log.info(f"Covid-19 Cases in Greece \n{df}")

# Convert data frame index to datetime index
log.info(f"data frame index: {df.index}")
time_series = df.reset_index()
log.info(f"data frame: \n{time_series.head()}")
# date_rng = pd.date_range(start='22/1/20', end='30/4/20')
date_rng = pd.date_range(start=str(time_series['Date'].iloc[0]), end=str(time_series['Date'].iloc[len(time_series)-1]))
time_series['DateTime'] = date_rng
log.info(f"data frame: \n{time_series.head()}")
time_series['DateTime'] = pd.to_datetime(time_series['DateTime'])
time_series = time_series.set_index('DateTime')
time_series = time_series.drop('Date', 1)
time_series.index.names = ['Date']
log.info(f"data frame: \n{time_series.head()}")
log.info(f"data frame index: {time_series.index}")

# Feature Engineering

# Calculate the number of new confirmed cases of every day
# in order not only to have the sum of all the case each day

# Initialize the list for the calculated feature New Confirmed Cases
new_confirmed_cases_list = list()
new_confirmed_cases_list.append(0)
log.info(f"new_confirmed_cases_list size before {len(new_confirmed_cases_list)}")

confirmed_cases_list = time_series['Confirmed Cases'].tolist()
log.info(f"confirmed_cases_list size: {len(confirmed_cases_list)}")

# Initialize the list for the calculated feature New Deaths
new_deaths_list = list()
new_deaths_list.append(0)
log.info(f"new_confirmed_cases_list size before {len(new_deaths_list)}")

deaths_list = time_series['Deaths'].tolist()
log.info(f"confirmed_cases_list size: {len(deaths_list)}")

# Initialize the list for the calculated feature New Recovered Cases
new_recovered_cases_list = list()
new_recovered_cases_list.append(0)
log.info(f"new_confirmed_cases_list size before {len(new_recovered_cases_list)}")

recovered_cases_list = time_series['Recovered Cases'].tolist()
log.info(f"recovered_cases_list size: {len(recovered_cases_list)}")

# Create features New Confirmed Cases, New Deaths and New Recovered Cases
if len(confirmed_cases_list) == len(deaths_list) == len(recovered_cases_list):
    i = 0
    for i in range(len(confirmed_cases_list)-1):
        new_cases = confirmed_cases_list[i+1] - confirmed_cases_list[i]
        new_confirmed_cases_list.append(new_cases)
        new_deaths = deaths_list[i+1] - deaths_list[i]
        new_deaths_list.append(new_deaths)
        new_recov = recovered_cases_list[i+1] - recovered_cases_list[i]
        new_recovered_cases_list.append(new_recov)

log.info(f"new_confirmed_cases_list size after {len(new_confirmed_cases_list)}")
log.info(f"new_confirmed_cases_list size after {len(new_deaths_list)}")
log.info(f"new_confirmed_cases_list size after {len(new_recovered_cases_list)}")

# Visualize the data at this point
time_series.plot(figsize=(15, 6))
plt.show()

time_series['New Confirmed Cases'] = new_confirmed_cases_list
time_series['New Deaths'] = new_deaths_list
time_series['New Recovered Cases'] = new_recovered_cases_list

new_time_series = time_series
new_time_series = new_time_series.drop(['Confirmed Cases', 'Deaths', 'Recovered Cases'], 1)
log.info(f"New Time Series Data Frame: \n{new_time_series.head()}")

# Visualize the data at this point
new_time_series.plot(figsize=(15, 6))
plt.show()

new_time_series['New Confirmed Cases'].plot(figsize=(15, 6))
plt.show()

new_time_series['New Deaths'].plot(figsize=(15, 6))
plt.show()

new_time_series['New Recovered Cases'].plot(figsize=(15, 6))
plt.show()

# Decomposition plots for:
# Confirmed Cases
x = time_series['New Confirmed Cases'].resample('D').mean()
rcParams['figure.figsize'] = 18, 8
decomposition = sm.tsa.seasonal_decompose(x, model='additive')
fig = decomposition.plot()
plt.show()
# Deaths
y = time_series['New Deaths'].resample('D').mean()
rcParams['figure.figsize'] = 18, 8
decomposition = sm.tsa.seasonal_decompose(y, model='additive')
fig = decomposition.plot()
plt.show()
# Recovered Cases
z = time_series['New Recovered Cases'].resample('D').mean()
rcParams['figure.figsize'] = 18, 8
decomposition = sm.tsa.seasonal_decompose(z, model='additive')
fig = decomposition.plot()
plt.show()

# Time Series Forecasting and Analysis

# Parameter Selection for ARIMA model
log.info("Parameter Selection for ARIMA model")
'''
ARIMA Parameters
p : seasonality
d : trend
q : noise
'''
p = d = q = range(0, 2)
pdq = list(itertools.product(p, d, q))
seasonal_pdq = [(x[0], x[1], x[2], 12) for x in list(itertools.product(p, d, q))]

for param in pdq:
    for param_seasonal in seasonal_pdq:
        try:
            mod = sm.tsa.statespace.SARIMAX(x, order=param, seasonal_order=param_seasonal, enforce_stationarity=False, enforce_invertibility=False)
            results = mod.fit()
        except:
            continue
    log.info('ARIMA{}x{}12 - AIC:{}'.format(param, param_seasonal, results.aic))

# Fit the ARIMA model
log.info("Fitting the ARIMA model")
mod = sm.tsa.statespace.SARIMAX(x, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12), enforce_stationarity=False, enforce_invertibility=False)
results = mod.fit()
# log.info(f"ARIMA Model Results: \n{results.get_forecast()}")
log.info(f"\n{results.summary().tables[1]}")

# Model Diagnostics
log.info("Model Diagnostics")
results.plot_diagnostics(figsize=(16, 8))
plt.show()

# Validating forecasts

pred = results.get_prediction(start=pd.to_datetime(str(time_series.reset_index()['Date'].iloc[0])), dynamic=False)
log.info(f"ARIMA Model Results: \n{pred}")
pred_ci = pred.conf_int()
log.info(f"ARIMA Model Results: \n{pred_ci}")
ax = x['2020-01-22':].plot(label='observed')
pred.predicted_mean.plot(ax=ax, label='One-step ahead Forecast', alpha=.7, figsize=(14, 7))
ax.fill_between(pred_ci.index, pred_ci.iloc[:, 0], pred_ci.iloc[:, 1], color='k', alpha=.2)
ax.set_xlabel('Date')
ax.set_ylabel('New Confirmed Cases')
plt.legend()
plt.show()

y_forecasted = pred.predicted_mean
y_truth = y['2017-01-01':]
mse = ((y_forecasted - y_truth) ** 2).mean()
log.info('The Mean Squared Error of our forecasts is {}'.format(round(mse, 2)))

log.info('The Root Mean Squared Error of our forecasts is {}'.format(round(np.sqrt(mse), 2)))

pred_uc = results.get_forecast(steps=100)
pred_ci = pred_uc.conf_int()
ax = y.plot(label='observed', figsize=(14, 7))
pred_uc.predicted_mean.plot(ax=ax, label='Forecast')
ax.fill_between(pred_ci.index,
                pred_ci.iloc[:, 0],
                pred_ci.iloc[:, 1], color='k', alpha=.25)
ax.set_xlabel('Date')
ax.set_ylabel('New Confirmed Cases')
plt.legend()
plt.show()

log.info("End of Execution\n\n\t\t\t\t------------ DATA MINING PIPELINE: END ------------\n")

# stop calculating execution time
end_time = time.time()
total_time = end_time - start_time
log.info(f"Execution Time: {total_time} seconds")

log.info("\n\n\t\t\t\t---- PROJECT: Predictions of Covid-19 in Greece ----"
         "\n\t\t\t\t----------------------------------------------------\n")
